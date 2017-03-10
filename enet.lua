local ct = 0
function _bottleneck(internal_scale, use_relu, asymetric, dilated, input, output, downsample)
   local internal = output / internal_scale
   local input_stride = downsample and 2 or 1

   local sum = nn.ConcatTable()

   local main = nn.Sequential()
   local other = nn.Sequential()
   sum:add(main):add(other)

   main:add(cudnn.SpatialConvolution(input, internal, input_stride, input_stride, input_stride, input_stride, 0, 0):noBias())
   main:add(nn.SpatialBatchNormalization(internal, 1e-3))
   if use_relu then main:add(nn.PReLU(internal)) end
   if not asymetric and not dilated then
      main:add(cudnn.SpatialConvolution(internal, internal, 3, 3, 1, 1, 1, 1))
   elseif asymetric then
      local pad = (asymetric-1) / 2
      main:add(cudnn.SpatialConvolution(internal, internal, asymetric, 1, 1, 1, pad, 0):noBias())
      main:add(cudnn.SpatialConvolution(internal, internal, 1, asymetric, 1, 1, 0, pad))
   elseif dilated then
      main:add(nn.SpatialDilatedConvolution(internal, internal, 3, 3, 1, 1, dilated, dilated, dilated, dilated))
   else
      assert(false, 'You shouldn\'t be here')
   end
   main:add(nn.SpatialBatchNormalization(internal, 1e-3))
   if use_relu then main:add(nn.PReLU(internal)) end
   main:add(cudnn.SpatialConvolution(internal, output, 1, 1, 1, 1, 0, 0):noBias())
   main:add(nn.SpatialBatchNormalization(output, 1e-3))
   main:add(nn.SpatialDropout((ct < 5) and 0.01 or 0.1))
   ct = ct + 1

   other:add(nn.Identity())
   if downsample then
      other:add(nn.SpatialMaxPooling(2, 2, 2, 2))
   end
   if input ~= output then
      other:add(nn.Padding(1, output-input, 3))
   end

   return nn.Sequential():add(sum):add(nn.CAddTable()):add(nn.PReLU(output))
end

function createModel(nGPU)
   local model = nn.Sequential()
   local _ = require 'moses'
   local bottleneck = _.bindn(_bottleneck, 4, true, false, false)
   local cbottleneck = _.bindn(_bottleneck, 4, true, false, false)
   local xbottleneck = _.bindn(_bottleneck, 4, true, 7, false)
   local wbottleneck = _.bindn(_bottleneck, 4, true, 5, false)
   local dbottleneck = _.bindn(_bottleneck, 4, true, false, 2)
   local xdbottleneck = _.bindn(_bottleneck, 4, true, false, 4)
   local xxdbottleneck = _.bindn(_bottleneck, 4, true, false, 8)
   local xxxdbottleneck = _.bindn(_bottleneck, 4, true, false, 16)
   local xxxxdbottleneck = _.bindn(_bottleneck, 4, true, false, 32)

   local initial_block = nn.ConcatTable(2)
   initial_block:add(cudnn.SpatialConvolution(3, 13, 3, 3, 2, 2, 1, 1))
   initial_block:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))

   model:add(initial_block) -- 112x112
   model:add(nn.JoinTable(2))
   model:add(nn.SpatialBatchNormalization(16, 1e-3))
   model:add(nn.PReLU(16))

   -- 1st block
   model:add(bottleneck(16, 64, true)) -- 56x56
   model:add(bottleneck(64, 128))
   model:add(bottleneck(128, 128))
   
   -- 2nd block: dilation of 2
   model:add(bottleneck(128, 256, true)) -- 28x28
   model:add(bottleneck(256, 256))
   model:add(dbottleneck(256, 256))

   -- 3rd block: dilation 4
   model:add(bottleneck(256, 512, true)) -- 14x14
   model:add(bottleneck(512, 512))
   model:add(xdbottleneck(512, 512))

   -- 4th block, dilation 8
   model:add(bottleneck(512, 1024, true)) -- 7x7
   model:add(bottleneck(1024, 1024))
   model:add(xxdbottleneck(1024, 1024))

   -- global average pooling 1x1
   model:add(cudnn.SpatialAveragePooling(7, 7, 1, 1, 0, 0))
   model:add(nn.LogSoftMax())
   model:cuda()
   model = makeDataParallel(model, nGPU) -- defined in util.lua
   model.imageSize = 256
   model.imageCrop = 224


   return model
end