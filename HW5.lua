-- Only requirements allowed
require("hdf5")
require("nn")
require("optim")

cmd = torch.CmdLine()

-- Cmd Args
cmd:option('-datafile', '', 'data file')
cmd:option('-classifier', 'hmm', 'classifier to use: hmm, memm, perceptron')
cmd:option('-model_out_name', 'train', 'output file name of model')
cmd:option('-action', 'train', 'train or test')
cmd:option('-test_model', '', 'model to test on')

-- Hyperparameters
cmd:option('-alpha', 0.01, 'smoothing alpha')
cmd:option('-beta', 1, 'beta for F-score')

cmd:option('-eta', 0.01, 'learning rate for SGD')
cmd:option('-batch_size', 32, 'batch size for SGD')
cmd:option('-max_epochs', 20, 'max # of epochs for SGD')

function get_context_features(X, Y)
  -- Change X, Y to context features for MEMM
  -- TODO: include more features
  local X_context = {} -- torch.Tensor(N-1, 2) -- x_i, c_{i-1}
  local Y_context = {}
  for j = 1, X:size(1) do
    local cur_X = X[j]
    local cur_Y = Y[j]
    for i = 2, X:size(2) do
      if cur_X[i] == end_word and cur_X[i] == cur_X[i-1] then break end
      table.insert(X_context, {cur_X[i], cur_Y[i-1] + vocab_size})
      table.insert(Y_context, cur_Y[i])
    end
  end
  
  return torch.LongTensor(X_context), torch.LongTensor(Y_context)
end

function strip_padding(X, pad)
  pad = pad or end_word

  -- handle padding
  local N = 1
  for i = 2, X:size(1) do
    if X[i] == X[i-1] and X[i] == pad then
      break
    end
    N = N + 1
  end
  return X:narrow(1, 1, N)
end

function get_hmm_probs(X, Y)
  local alpha = opt.alpha

  local N = X:size(1)
  local transitions = torch.Tensor(nclasses, nclasses):fill(alpha)
  local emissions = torch.Tensor(nfeatures, nclasses):fill(alpha)
  for j = 1, N do
    local seq = X[j]
    local lbl = Y[j]
    for i = 1, seq:size(1) do
      if i > 1 then
        if lbl[i] == lbl[i-1] and lbl[i] == end_tag then
          -- end of sentence
          break
        end
        transitions[lbl[i-1]][lbl[i]] = transitions[lbl[i-1]][lbl[i]] + 1
      end
      emissions[seq[i]][lbl[i]] = emissions[seq[i]][lbl[i]] + 1
    end
  end

  transitions:cdiv(transitions:sum(2):expand(nclasses, nclasses))
  transitions:log()
  emissions:cdiv(emissions:sum(2):expand(nfeatures, nclasses))
  emissions:log()

  return transitions, emissions
end

function viterbi(X, transitions, emissions, model)
  -- handle padding
  X = strip_padding(X)
  local N = X:size(1)
  local pi = torch.Tensor(N, nclasses):fill(-math.huge)
  local bp = torch.zeros(N, nclasses):long()
  local cache = {}

  -- initialize
  if model then
    pi[1]:zero()
  elseif transitions then
    pi[1] = emissions[X[1]]
  end

  for i = 2, N do
    for prev_c = 1, nclasses do
      local y_hat
      if model then
        -- batch size 1 for model compatibility
        y_hat = model:forward(torch.LongTensor{{X[i], prev_c + vocab_size}})
        y_hat = y_hat:squeeze()
      elseif transitions then
        y_hat = emissions[X[i]] + transitions[prev_c]
      end
      for c = 1, nclasses do
        if pi[i-1][prev_c] + y_hat[c] > pi[i][c] then
          pi[i][c] = pi[i-1][prev_c] + y_hat[c]
          bp[i][c] = prev_c
        end
      end
    end
  end

  -- trace backpointers
  local _, p = torch.max(pi[N], 1)
  p = p[1]
  local seq = {}
  table.insert(seq, p)
  for i = N, 2, -1 do
    p = bp[i][p]
    table.insert(seq, p)
  end

  -- reverse
  local rev_seq = torch.LongTensor(N)
  for i = N, 1, -1 do
    rev_seq[i] = seq[N-i+1]
  end

  return rev_seq
end

function compute_fscore(total_predicted_correct, total_predicted, total_correct, beta)
  beta = beta or opt.beta

  local prec = total_predicted_correct / total_predicted
  local rec = total_predicted_correct / total_correct
  return (beta * beta + 1) * prec * rec / (beta * beta * prec + rec)
end

function get_mentions(seq)
  local N = seq:size(1)
  local seq_mentions = {}

  local cur_mention = seq[2]
  local cur_start = 2
  for i = 3, N-1 do
    -- transition to new mention.
    if seq[i] ~= cur_mention then
      local str
      if cur_start == i-1 then str = tostring(i-2) else str = (cur_start-1) .. '-' .. (i-2) end
      if seq_mentions[cur_mention] == nil then
        seq_mentions[cur_mention] = {}
        seq_mentions[cur_mention][str] = true
      else
        seq_mentions[cur_mention][str] = true
      end
      cur_mention = seq[i]
      cur_start = i
    end
  end
  
  seq_mentions[1] = nil -- no need for O mentions
  return seq_mentions
end

function score_seq(seq, Y)
  assert(Y:size(1) == seq:size(1))
  local pred_correct = 0
  local pred = 0
  local correct = 0
      
  local seq_mentions = get_mentions(seq)
  local Y_mentions = get_mentions(Y)

  for name,m in pairs(seq_mentions) do
    for k,_ in pairs(m) do
      if Y_mentions[name] ~= nil and Y_mentions[name][k] ~= nil then
        pred_correct = pred_correct + 1
      end
      pred = pred + 1
    end
  end
  for _,m in pairs(Y_mentions) do
    for k,_ in pairs(m) do
      correct = correct + 1
    end
  end

  --print(seq_mentions)
  --print(Y_mentions)
  --print(pred_correct, pred, correct)
  --io.read()
  return pred_correct, pred, correct
end

function compute_eval_err(X, Y, transitions, emissions, model)
   -- predicted entities
   local tot_pred_correct = 0
   local tot_pred = 0
   local tot_correct = 0
   for i = 1, X:size(1) do
     local seq = viterbi(X[i], transitions, emissions, model)
     local Y_seq = strip_padding(Y[i], end_tag)

     local pred_correct, pred, correct = score_seq(seq, Y_seq)
     tot_pred_correct = tot_pred_correct + pred_correct
     tot_pred = tot_pred + pred
     tot_correct = tot_correct + correct
   end

   local fscore = compute_fscore(tot_pred_correct, tot_pred, tot_correct)
   return fscore
 end

function MEMM()
  local model = nn.Sequential()
  model:add(nn.LookupTable(nfeatures + nclasses, nclasses))
  model:add(nn.Sum(2)) -- sum w_i*x_i
  model:add(nn.Add(nclasses)) -- bias
  model:add(nn.LogSoftMax())
  
  return model
end

function perceptron()
  local model = nn.Sequential()
  model:add(nn.LookupTable(nfeatures, nclasses))
  -- no softmax or bias (?)

  return model
end

function model_eval(model, criterion, X, Y)
  -- batch eval
  model:evaluate()
  local N = X:size(1)
  local batch_size = opt.batch_size

  local total_loss = 0
  for batch = 1, N, batch_size do
      local sz = batch_size
      if batch + batch_size > N then
        sz = N - batch + 1
      end
      local X_batch = X:narrow(1, batch, sz)
      local Y_batch = Y:narrow(1, batch, sz)

      local inputs = X_batch
      local outputs = model:forward(inputs)
      local loss = criterion:forward(outputs, Y_batch)

      total_loss = total_loss + loss * batch_size
  end

  return total_loss / N
end

function train_model(X, Y, valid_X, valid_Y)
  local eta
  if opt.classifier == 'memm' then
    eta = opt.eta
  elseif opt.classifier == 'perceptron' then
    eta = 1
  end
  local batch_size = opt.batch_size
  local max_epochs = opt.max_epochs
  local N = X:size(1)

  local model
  if opt.classifier == 'memm' then
    model = MEMM()
  elseif opt.classifier == 'perceptron' then
    model = perceptron()
  end

  local criterion = nn.ClassNLLCriterion()

  -- shuffle for batches
  local shuffle = torch.randperm(N):long()
  X = X:index(1, shuffle)
  Y = Y:index(1, shuffle)

  -- only call this once
  local params, grads = model:getParameters()
  local state = { learningRate = eta }

  local prev_loss = 1e10
  local epoch = 1
  local timer = torch.Timer()
  while epoch <= max_epochs do
      print('Epoch:', epoch)
      local epoch_time = timer:time().real
      local total_loss = 0

      -- loop through each batch
      model:training()
      for batch = 1, N, batch_size do
          --if ((batch - 1) / batch_size) % 1000 == 0 then
            --print('Sample:', batch)
            --print('Current train loss:', total_loss / batch)
            --print('Current time:', 1000 * (timer:time().real - epoch_time), 'ms')
          --end
          local sz = batch_size
          if batch + batch_size > N then
            sz = N - batch + 1
          end
          local X_batch = X:narrow(1, batch, sz)
          local Y_batch = Y:narrow(1, batch, sz)

          local func
          if opt.classifier == 'memm' then
            -- closure to return err, df/dx
            func = function(x)
              -- get new parameters
              if x ~= params then
                params:copy(x)
              end
              -- reset gradients
              grads:zero()

              -- forward
              local inputs = X_batch
              local outputs = model:forward(inputs)
              local loss = criterion:forward(outputs, Y_batch)

              -- track errors
              total_loss = total_loss + loss * batch_size

              -- compute gradients
              local df_do = criterion:backward(outputs, Y_batch)
              model:backward(inputs, df_do)

              return loss, grads
            end
          elseif opt.classifier == 'perceptron' then
            func = function(x)
              -- get new parameters
              if x ~= params then
                params:copy(x)
              end
              -- reset gradients
              grads:zero()

              -- forward
              local inputs = X_batch

              -- do Viterbi on each input with this model
              -- diff each result with Y_batch (gold) to get grads
              -- recompute (?) and then construct grads
              -- backward

              return loss, grads
            end

            -- maybe manually sgd
          end

          optim.sgd(func, params, state)
      end

      print('Train loss:', total_loss / N)

      local loss = model_eval(model, criterion, valid_X, valid_Y)
      print('Valid loss:', loss)
      -- Viterbi sequence
      --local seq = viterbi(X, nil, nil, model)
      --local fscore = score_seq(seq, Y)
      --print('Valid F-score:', fscore)

      print('time for one epoch: ', (timer:time().real - epoch_time) * 1000, 'ms')
      print('')
      if loss > prev_loss and epoch > 5 then
        prev_loss = loss
        break
      end
      prev_loss = loss
      epoch = epoch + 1
      torch.save(opt.model_out_name .. '_' .. opt.classifier .. '.t7', { model = model })
  end
  print('Trained', epoch-1, 'epochs')
  return model, prev_loss
end

function main() 
   -- Parse input params
   opt = cmd:parse(arg)
   local f = hdf5.open(opt.datafile, 'r')
   local X = f:read('train_input'):all():long()
   local Y = f:read('train_output'):all():long()
   local valid_X = f:read('valid_input'):all():long()
   local valid_Y = f:read('valid_output'):all():long()
   local test_X = f:read('test_input'):all():long()
   --local test_ids = f:read('test_ids'):all():long()
   -- Word embeddings from glove
   --local word_vecs = f:read('word_vecs'):all()
   --vec_size = word_vecs:size(2)

   nclasses = f:read('nclasses'):all():long()[1]
   nfeatures = f:read('nfeatures'):all():long()[1]
   vocab_size = f:read('vocab_size'):all():long()[1]
   start_word = 1
   end_word = 2
   start_tag = 8
   end_tag = 9
   tags = {'O', 'I-PER', 'I-LOC', 'I-ORG', 'I-MISC', 'B-MISC', 'B-LOC'}

   if opt.classifier == 'hmm' then
     local transitions, emissions = get_hmm_probs(X, Y)
     print('Computed HMM probs')

     local timer = torch.Timer()
     local time = timer:time().real
     local fscore = compute_eval_err(valid_X, valid_Y, transitions, emissions, nil)
     print('Valid time:', (timer:time().real - time) * 1000, 'ms')
     print('Valid F-score:', fscore)
   elseif opt.classifier == 'memm' then
     local model
     if opt.action == 'train' then
       -- context for training
       local X_context, Y_context = get_context_features(X, Y)
       local valid_X_context, valid_Y_context = get_context_features(valid_X, valid_Y)
       model = train_model(X_context, Y_context, valid_X_context, valid_Y_context)
     else
       model = torch.load(opt.test_model).model
     end

     -- Viterbi
     local timer = torch.Timer()
     local time = timer:time().real
     local fscore = compute_eval_err(valid_X, valid_Y, nil, nil, model)
     print('Valid time:', (timer:time().real - time) * 1000, 'ms')
     print('Valid F-score:', fscore)
   elseif opt.classifier == 'perceptron' then
     -- a???
   end
end

main()
