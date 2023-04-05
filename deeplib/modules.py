
from deeplib.engine import Tensor
    

class Module:
    
    def __init__(self) -> None:
        self.modules = []
        
    def __call__(self, batch:Tensor) -> Tensor:
        outputs = []
        for sample in batch:
            out = self.forward(sample)
            outputs.append(out)     
        return Tensor.concat(outputs, dim=0)

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()
    
    def freeze(self):
        for p  in self.parameters():
            p.requires_grad = False
    
    def unfreeze(self):
        for p  in self.parameters():
            p.requires_grad = True
            
    def track_module(self, m:'Module'):
        self.modules.append(m)
        
    def track_modules(self, modules:list['Module']):
        self.modules += modules

    def parameters(self) -> list['Tensor']:
        return [p for m in self.modules for p in m.parameters()]
    
    def num_params(self, trainable=False, non_trainable=False) -> int:
        num_params = 0; num_trainable = 0; num_non_trainable = 0
        for p in self.parameters():
            num_params += p.size
            if p.requires_grad: num_trainable += p.size
            else: num_non_trainable += p.size
        
        if trainable: return num_trainable
        elif non_trainable: return num_non_trainable
        return num_params
        
    def forward(self, x:Tensor) -> Tensor:
        pass
    
    def __repr__(self) -> str:
        return (f"Module(tracked_modules={len(self.modules)}, " +
                f"parameters={self.num_params()}, trainable={self.num_params(trainable=True)}, " +
                f"non_trainable={self.num_params(non_trainable=True)})")
        
        
class Sequential(Module):
    
    def __init__(self, *modules) -> None:
        super().__init__()
        self.track_modules(modules)
        
    def forward(self, x: Tensor) -> Tensor:
        inp = x
        for module in self.modules:
            out = module(inp)
            inp = out
        return out
        
