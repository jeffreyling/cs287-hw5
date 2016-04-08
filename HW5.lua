-- Only requirements allowed
require("hdf5")
require("nn")
require("optim")

cmd = torch.CmdLine()

-- Cmd Args
cmd:option('-datafile', '', 'data file')
cmd:option('-classifier', 'hmm', 'classifier to use: hmm, memm')

-- Hyperparameters
cmd:option('-alpha', 0.1, 'smoothing alpha')
cmd:option('-beta', 1, 'beta for F-score')

function get_hmm_probs(X, Y)
  local alpha = opt.alpha

  local N = X:size(1)
  local transitions = torch.Tensor(nclasses, nclasses):fill(alpha)
  local emissions = torch.Tensor(nfeatures, nclasses):fill(alpha)
  for i = 1, N do
    if i > 1 then
      transitions[Y[i-1]][Y[i]] = transitions[Y[i-1]][Y[i]] + 1
    end
    emissions[X[i]][Y[i]] = emissions[X[i]][Y[i]] + 1
  end

  transitions:div(transitions:sum(2):expand(nclasses, nclasses))
  transitions:log()
  emissions:div(emissions:sum(2):expand(nfeatures, nclasses))
  emissions:log()

  return transitions, emissions
end

function viterbi(X, transitions, emissions, model)
  -- use transition/emission probs or model
  local N = X:size(1)
  local pi = torch.Tensor(N, nclasses):fill(-math.huge)
  local bp = torch.zeros(N, nclasses):long()

  -- initialize
  if model then
    pi[1]:zero()
  elseif transitions then
    for c = 1, nclasses do
      pi[1][c] = emissions[X[1]][c]
    end
  end

  for i = 2, N do
    for prev_c = 1, nclasses do
      local y_hat
      if model then
        -- ????
        -- y_hat = model:forward({X[i], prev_c})
        pass
      elseif transitions then
        y_hat = emissions[X[i]] + transitions[prev_c]
      end
      for c = 1, nclasses do
        if pi[i-1][c] + y_hat[c] > pi[i][c] then
          pi[i][c] = pi[i-1][c] + y_hat
          bp[i][c] = prev_c
        end
      end
    end
  end

  -- trace backpointers
  local _, p = torch.max(pi[N], 1)
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

function MEMM()
  local model = nn.Sequential()
  model:add(nn.LookupTable(nfeatures, nclasses))
  model:add(nn.Add(nclasses)) -- bias
  model:add(nn.LogSoftMax())
  
  return model
end

function compute_fscore(total_predicted_correct, total_predicted, total_correct, beta)
  local prec = total_predicted_correct / total_predicted
  local rec = total_predicted_correct / total_correct
  return (beta * beta + 1) * prec * rec / (beta * beta * prec + rec)
end

function score_seq(seq, Y)
  local N = seq:size(1)
  assert(Y:size(1) == N)
  local total_predicted_correct = seq:eq(Y):sum()
  return compute_fscore(total_predicted_correct, N, N, opt.beta)
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

  return loss
end

function train_model(X, Y, valid_X, valid_Y)
  local eta = opt.eta
  local batch_size = opt.batch_size
  local max_epochs = opt.max_epochs
  local N = X:size(1)

  local model
  if opt.classifier == 'memm' then
    model = MEMM()
  else
    -- asdf
    asdf
  end

  local criterion = nn.ClassNLLCriterion()

  -- shuffle for batches
  local shuffle = torch.randperm(N):long()
  X = X:index(1, shuffle)
  -- features X?
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
          --local X_cap_batch = X_cap:narrow(1, batch, sz)
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
   local valid_Y = f:read('valid_output'):all()
   local test_X = f:read('test_input'):all():long()
   --local test_ids = f:read('test_ids'):all():long()
   -- Word embeddings from glove
   --local word_vecs = f:read('word_vecs'):all()
   --vec_size = word_vecs:size(2)

   nclasses = f:read('nclasses'):all():long()[1]
   nfeatures = f:read('nfeatures'):all():long()[1]

   if opt.classifier == 'hmm' then
     local transitions, emissions = get_hmm_probs(X, Y)

     local valid_seq = viterbi(valid_X, transitions, emissions, nil)
     local fscore = score_seq(valid_seq, valid_Y)
     print('Valid F-score:', fscore)
   elseif opt.classifier == 'memm' then
     local model = train_model(X, Y, valid_X, valid_Y)

     -- Viterbi sequence
     local seq = viterbi(X, nil, nil, model)
     local fscore = score_seq(seq, Y)
     print('Valid F-score:', fscore)
   end
end

main()
