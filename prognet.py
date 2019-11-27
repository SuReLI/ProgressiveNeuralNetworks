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
                out.append(col[l](inputs[:i+1],activated = (l== self.depth - 1)))

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
