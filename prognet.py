import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
from operator import mul
from collections import Iterable

class LateralBlock(nn.Module):
    def __init__(self,col,depth,block,out_shape, in_shapes):
        super(LateralBlock,self).__init__()
        self.col = col
        self.depth = depth
        self.out_shape = out_shape
        self.block = block
        self.u = nn.ModuleList()


        if self.depth > 0:
            red_in_shapes = [reduce(mul,in_shape) if isinstance(in_shape,Iterable) else in_shape for in_shape in in_shapes]
            red_out_shape = reduce(mul,out_shape) if isinstance(out_shape,Iterable) else out_shape
            self.u.extend([nn.Linear(in_shape,red_out_shape) for in_shape in red_in_shapes])


    def forward(self,inputs,activated = True):
        if not isinstance(inputs, list):
            inputs = [inputs]

        cur_column_out = self.block(inputs[-1])
        out_shape = tuple(j for i in (-1, self.out_shape) for j in (i if isinstance(i, tuple) else (i,)))
        prev_columns_out = [mod(x.view(x.shape[0],-1)).view(out_shape) for mod, x in zip(self.u, inputs)]
        res= cur_column_out + sum(prev_columns_out)
        if activated:
            res = F.relu(res)
        return res



class ProgNet(nn.Module):
    def __init__(self,depth):
        super(ProgNet,self).__init__()

        self.columns = nn.ModuleList([])
        self.depth = depth


    def forward(self,x,task_id=-1):
        assert self.columns
        inputs = [col[0](x) for col in self.columns]
        for l in range(1,self.depth):
            out = []
            for i,col in enumerate(self.columns):
                out.append(col[l](inputs[:i+1],activated = (l!= self.depth - 1)))

            inputs = out
        return out[task_id]

    def new_task(self,new_layers,shapes):
        assert isinstance(new_layers,nn.Sequential)
        assert(len(new_layers) == len(shapes))

        task_id = len(self.columns)
        idx =[i for i,layer in enumerate(new_layers) if isinstance(layer,(nn.Conv2d,nn.Linear))] + [len(new_layers)]
        new_blocks = []

        for k in range(len(idx) -1):
            prev_blocks = []
            if k > 0:
                prev_blocks = [col[k-1] for col in self.columns]

            new_blocks.append(LateralBlock(col = task_id,
                                           depth = k,
                                           block = new_layers[idx[k]:idx[k+1]],
                                           out_shape = shapes[idx[k+1]-1],
                                           in_shapes = self._get_out_shape_blocks(prev_blocks)
                                          ))

        new_column = nn.ModuleList(new_blocks)
        self.columns.append(new_column)



    def _get_out_shape_blocks(self,blocks):
        assert isinstance(blocks,list)
        assert all(isinstance(block,LateralBlock) for block in blocks)
        return [block.out_shape for block in blocks]



    def freeze_columns(self, skip=None):
        if skip == None:
            skip = []

        for i, c in enumerate(self.columns):
            if i not in skip:
                for params in c.parameters():
                    params.requires_grad = False


class LateralBlockSimple(nn.Module):
    def __init__(self,col,depth,block,out_shape, in_shapes):
        super(LateralBlockSimple,self).__init__()
        self.col = col
        self.depth = depth
        self.out_shape = out_shape
        self.block = block
        self.u = nn.ModuleList()
        self.fp = 0
        if self.depth > 0:
            self.u.extend([nn.Linear(in_shape,out_shape) for in_shape in in_shapes])


    def forward(self,inputs,activated = True):
        if not isinstance(inputs, list):
            inputs = [inputs]

        cur_column_out = self.block(inputs[-1])
        prev_columns_out = [mod(x) for mod, x in zip(self.u, inputs)]
        res= cur_column_out + sum(prev_columns_out)

        if activated:
            res = F.relu(res)
        self.fp += 1
        return res

    def forward_skip(self,inputs,block_mask,activated = True):
        if not isinstance(inputs, list):
            inputs = [inputs]

        cur_column_out = self.block(inputs[-1])
        if block_mask is None:
            prev_columns_out = [mod(x) for mod, x in zip(self.u, inputs)]
            res = cur_column_out + sum(prev_columns_out)

        else:
            prev_columns_out = [block_mask[i]*mod(x) for i,(mod, x) in enumerate(zip(self.u, inputs))]
            res= block_mask[-1]*cur_column_out + sum(prev_columns_out)

        if activated:
            res = F.relu(res)
        self.fp += 1
        return res

class ProgNetSimple(nn.Module):
    def __init__(self):
        super(ProgNetSimple,self).__init__()

        self.columns = nn.ModuleList([])

    def forward(self,x,task_id=-1):
        assert self.columns
        inputs = [col[0](x) for col in self.columns]
        for l in range(1,self.depth):
            out = []
            for i,col in enumerate(self.columns):
                out.append(col[l](inputs[:i+1],activated = (l!= self.depth - 1)))

            inputs = out
        return out[task_id]

    def forward_skip(self,x,blocks_mask,task_id=-1):
         assert self.columns
         assert (blocks_mask.shape[0] == self.depth-1)
         assert (blocks_mask.shape[1] == task_id%len(self.columns) + 1)

         inputs = [col[0](x) for col in self.columns]
         for l in range(1,self.depth):
             out = []
             for i,col in enumerate(self.columns):
                 out.append(col[l].forward_skip(inputs[:i+1],
                                   block_mask = blocks_mask[l-1]  if (i == task_id%len(self.columns)) else None,
                                   activated = (l!= self.depth - 1),
                                   ))

             inputs = out
         return out[task_id]


    def new_task(self,in_size,out_shapes):
        if not hasattr(self,'in_size'):
            self.in_size = in_size
        if not hasattr(self,'depth'):
            self.depth = len(out_shapes)

        assert out_shapes
        assert(self.in_size == in_size)
        assert(self.depth == len(out_shapes))

        task_id = len(self.columns)
        new_blocks = []
        new_layers = [nn.Linear(self.in_size, out_shapes[0])] + [nn.Linear(out_shapes[k-1],out_shapes[k]) for k in range(1,self.depth)]
        idx = [i for i in range(len(new_layers) + 1)]

        for k in range(len(idx) -1):
            prev_blocks = []
            if k > 0:
                prev_blocks = [col[k-1] for col in self.columns]

            new_blocks.append(LateralBlockSimple(col = task_id,
                                           depth = k,
                                           block = new_layers[k],
                                           out_shape = out_shapes[k],
                                           in_shapes = self._get_out_shape_blocks(prev_blocks),
                                          ))

        new_column = nn.ModuleList(new_blocks)
        self.columns.append(new_column)



    def _get_out_shape_blocks(self,blocks):
        assert isinstance(blocks,list)
        assert all(isinstance(block,LateralBlockSimple) for block in blocks)
        return [block.out_shape for block in blocks]



    def freeze_columns(self, skip=None):
        if skip == None:
            skip = []

        for i, c in enumerate(self.columns):
            if i not in skip:
                for params in c.parameters():
                    params.requires_grad = False
