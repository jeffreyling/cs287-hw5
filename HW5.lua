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
cmd:option('-warm_start_model', '', 'model to restart training')

-- Hyperparameters
cmd:option('-alpha', 0.01, 'smoothing alpha')
cmd:option('-beta', 1, 'beta for F-score')

cmd:option('-eta', 0.01, 'learning rate for SGD')
cmd:option('-batch_size', 32, 'batch size for SGD')
cmd:option('-max_epochs', 20, 'max # of epochs for SGD')

function to_feats(c, x, x_feats)
  -- batch size 1 for model compatibility
  local words = x:view(1, x:size(1))
  local all_feats = torch.cat(torch.LongTensor{c + nfeatures*window_size}, x_feats, 1)
  all_feats = all_feats:view(1, all_feats:size(1))
  return {words, all_feats}
end

function hash(feats)
  -- hash for viterbi speedup
  local h = ''
  for _,v in ipairs(feats) do
    for i = 1, v[1]:size(1) do
      h = h .. ',' .. v[1][i]
    end
  end
  return h
end

function strip_padding(X, pad)
  -- remove padding for a sentence
  pad = pad or end_word

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
  local emissions = torch.Tensor(vocab_size, nclasses):fill(alpha)
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
  emissions:cdiv(emissions:sum(2):expand(vocab_size, nclasses))
  emissions:log()

  return transitions, emissions
end

function init_window(X)
  local w = math.floor(window_size / 2)
  local pad_X = torch.cat(torch.LongTensor(w):fill(start_word), X, 1)
  pad_X = torch.cat(pad_X, torch.LongTensor(w):fill(end_word), 1)

  local X_window = torch.LongTensor(X:size(1), window_size)
  for i = 1, X:size(1) do
    X_window[i] = pad_X:narrow(1, i, window_size)
  end

  return X_window
end

function init_window_feats(X)
  local w = math.floor(window_size / 2)
  local num_feats = X:size(2)
  local pad_X = torch.cat(torch.LongTensor(w, num_feats):fill(start_word), X, 1)
  pad_X = torch.cat(pad_X, torch.LongTensor(w, num_feats):fill(end_word), 1)

  local X_window = torch.LongTensor(X:size(1), num_feats * window_size)
  for i = 1, X:size(1) do
    X_window[i] = pad_X:narrow(1, i, window_size):view(num_feats * window_size)
  end

  return X_window
end

function viterbi(X, transitions, emissions, model, X_feats)
  -- handle padding
  X = strip_padding(X)
  X_feats = X_feats:narrow(1, 1, X:size(1))
  local N = X:size(1)
  local pi = torch.Tensor(N, nclasses):fill(-math.huge)
  local bp = torch.zeros(N, nclasses):long()
  --local cache = {}

  -- initialize
  local X_window
  local X_feats_window
  if model then
    pi[1]:zero()

    -- initialize windows
    X_window = init_window(X)
    X_feats_window = init_window_feats(X_feats)
  elseif transitions then
    pi[1] = emissions[X[1]]
  end

  for i = 2, N do
    for prev_c = 1, nclasses do
      local y_hat
      if model then
        local feats = to_feats(prev_c, X_window[i], X_feats[i])
        --local h = hash(feats)
        --if cache[h] then
          --y_hat = cache[h]
        --else
          y_hat = model:forward(feats)
          y_hat = y_hat:squeeze()
          --cache[h] = y_hat:clone()
        --end
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
  -- Gets table of mentions from a sequence
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
  -- compares pred vs gold sequence and output mention counts
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

function compute_eval_err(X, Y, transitions, emissions, model, X_feats)
   -- Compute F score of predicted mentions
   local tot_pred_correct = 0
   local tot_pred = 0
   local tot_correct = 0
   for i = 1, X:size(1) do
     local seq
     if model then
       seq = viterbi(X[i], nil, nil, model, X_feats[i])
     elseif transitions then
       seq = viterbi(X[i], transitions, emissions, nil, nil)
     end
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
  if opt.warm_start_model ~= '' then
    return torch.load(opt.warm_start_model).model
  end

  local model = nn.Sequential()
  local inputs = nn.ParallelTable()
  local word_lookup = nn.LookupTable(vocab_size * window_size, nclasses)
  local feats_lookup = nn.LookupTable(nfeatures * window_size + nclasses, nclasses)
  inputs:add(word_lookup)
  inputs:add(feats_lookup)
  model:add(inputs)
  model:add(nn.JoinTable(2))
  model:add(nn.Sum(2)) -- sum w_i*x_i
  model:add(nn.Add(nclasses)) -- bias
  model:add(nn.LogSoftMax())
  
  return model
end

function perceptron()
  local model = nn.Sequential()
  local lookup = nn.LookupTable(nfeatures + nclasses, nclasses)
  lookup.weight:zero()
  model:add(lookup)
  model:add(nn.Sum(2))
  -- no softmax or bias

  return model
end

function model_eval(model, criterion, X, Y, X_feats)
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
      local X_feats_batch = X_feats:narrow(1, batch, sz)

      local inputs = {X_batch, X_feats_batch}
      local outputs = model:forward(inputs)
      local loss = criterion:forward(outputs, Y_batch)

      total_loss = total_loss + loss * batch_size
  end

  return total_loss / N
end

function train_perceptron(X, Y, valid_X, valid_Y)
  -- X, Y in sentence format
  local max_epochs = opt.max_epochs
  local N = X:size(1)

  local model = perceptron()

  -- shuffle for batches
  local shuffle = torch.randperm(N):long()
  X = X:index(1, shuffle)
  Y = Y:index(1, shuffle)

  local prev_loss = 1e10
  local epoch = 1
  local timer = torch.Timer()
  while epoch <= max_epochs do
      print('Epoch:', epoch)
      local epoch_time = timer:time().real
      local total_loss = 0

      model:training()
      model:zeroGradParameters()

      -- do Viterbi on each input
      for i = 1, X:size(1) do
        local seq = viterbi(X[i], nil, nil, model)
        local Y_seq = strip_padding(Y[i], end_tag)
        assert(seq:size(1) == Y_seq:size(1))
        for k = 2, seq:size(1) do
          if seq[k] ~= Y_seq[k] then
            -- do update where pred does not equal gold
            model:zeroGradParameters()
            local input = to_feats(seq[k-1], X[i][k])
            local y_hat = model:forward(input)
            local _,m = torch.max(y_hat, 1)
            m = torch.min(m[1])

            -- construct gradient
            local g = torch.zeros(1, nclasses)
            g[1][Y_seq[k]] = -1
            g[1][m] = 1
            model:backward(input, g)
            model:updateParameters(1)
          end
        end
      end

      -- evaluate
      local fscore = compute_eval_err(valid_X, valid_Y, nil, nil, model)
      print('Valid F-score:', fscore)
      local loss = 1 - fscore

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

function train_model(X, Y, X_feats, valid_X, valid_Y, valid_X_feats)
  local eta = opt.eta
  local batch_size = opt.batch_size
  local max_epochs = opt.max_epochs
  local N = X:size(1)

  local model = MEMM()
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
          local sz = batch_size
          if batch + batch_size > N then
            sz = N - batch + 1
          end
          local X_batch = X:narrow(1, batch, sz)
          local X_feats_batch = X_feats:narrow(1, batch, sz)
          local Y_batch = Y:narrow(1, batch, sz)

          -- closure to return err, df/dx
          local func = function(x)
            -- get new parameters
            if x ~= params then
              params:copy(x)
            end
            -- reset gradients
            grads:zero()

            -- forward
            local inputs = {X_batch, X_feats_batch}
            local outputs = model:forward(inputs)
            local loss = criterion:forward(outputs, Y_batch)

            -- track errors
            total_loss = total_loss + loss * batch_size

            -- compute gradients
            local df_do = criterion:backward(outputs, Y_batch)
            model:backward(inputs, df_do)

            return loss, grads
          end

          optim.sgd(func, params, state)
      end

      print('Train loss:', total_loss / N)
      local loss = model_eval(model, criterion, valid_X, valid_Y, valid_X_feats)
      print('Valid loss:', loss)

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
   --
   -- More features
   local X_feats = f:read('train_features_input'):all():long()
   local valid_X_feats = f:read('valid_features_input'):all():long()

   -- window format for MEMM
   local X_window = f:read('train_input_window'):all():long()
   local X_feats_window = f:read('train_feats_input_window'):all():long()
   local Y_window = f:read('train_output_window'):all():long()
   local valid_X_window = f:read('valid_input_window'):all():long()
   local valid_X_feats_window = f:read('valid_feats_input_window'):all():long()
   local valid_Y_window = f:read('valid_output_window'):all():long()

   nclasses = f:read('nclasses'):all():long()[1]
   nfeatures = f:read('nfeatures'):all():long()[1]
   vocab_size = f:read('vocab_size'):all():long()[1]
   start_word = 1
   end_word = 2
   start_tag = 8
   end_tag = 9
   window_size = 5 -- set for now
   --tags = {'O', 'I-PER', 'I-LOC', 'I-ORG', 'I-MISC', 'B-MISC', 'B-LOC'}

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
       model = train_model(X_window, Y_window, X_feats_window, valid_X_window, valid_Y_window, valid_X_feats_window)
     else
       model = torch.load(opt.test_model).model
     end

     -- Viterbi
     local timer = torch.Timer()
     local time = timer:time().real
     local fscore = compute_eval_err(valid_X, valid_Y, nil, nil, model, X_feats)
     print('Valid time:', (timer:time().real - time) * 1000, 'ms')
     print('Valid F-score:', fscore)
   elseif opt.classifier == 'perceptron' then
     local model = train_perceptron(X, Y, valid_X, valid_Y)
   end
end

main()
