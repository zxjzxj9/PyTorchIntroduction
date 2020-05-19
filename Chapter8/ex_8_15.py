""" 本代码仅供参考
"""

# jit.trace函数的签名
torch.jit.trace(func, example_inputs, optimize=None, 
                check_trace=True, check_inputs=None, check_tolerance=1e-5)

def func(a):
    return a.pow(2) + 1

class Mod(nn.Module):
    def __init__(self):
        super(Mod, self).__init__()

    def forward(self, a):
        return a.pow(2) + 1

ret = torch.jit.trace(func, torch.randn(3,3))
print(ret.graph)
# 打印出的值：
# graph(%a : Float(3, 3)):
#   %1 : int = prim::Constant[value=2]() 
#   %2 : Float(3, 3) = aten::pow(%a, %1)
#   %3 : Long() = prim::Constant[value={1}]()
#   %4 : int = prim::Constant[value=1]() 
#   %5 : Float(3, 3) = aten::add(%2, %3, %4) 
#  return (%5)
m = Mod()
ret = torch.jit.trace(m, torch.randn(3,3))
print(ret.graph)
# 打印出的值：
# graph(%self : ClassType<Mod>,
#       %a : Float(3, 3)):
#   %2 : int = prim::Constant[value=2](), scope: Mod #
#   %3 : Float(3, 3) = aten::pow(%a, %2), scope: Mod #
#   %4 : Long() = prim::Constant[value={1}](), scope: Mod
#   %5 : int = prim::Constant[value=1](), scope: Mod
#   %6 : Float(3, 3) = aten::add(%3, %4, %5), scope: Mod
#   return (%6)

# jit.trace_module函数的签名
torch.jit.trace_module(mod, inputs, optimize=None, check_trace=True,
                       check_inputs=None, check_tolerance=1e-5)

class Mod(nn.Module):
    def __init__(self):
        super(Mod, self).__init__()

    def forward(self, a):
        return a.pow(2) + 1
    
    def square(self, a):
        return a.pow(2)

trace_input = {"forward": torch.randn(3,3), "square": torch.randn(3,3)}
m = Mod()
ret = torch.jit.trace_module(m, trace_input)
print(ret.forward.graph) # 和前面的torch.jit.trace函数输出的结果相同
print(ret.square.graph)
# 打印出的值：
# graph(%self : ClassType<Mod>,
#       %a : Float(3, 3)):
#  %2 : int = prim::Constant[value=2]() #  %3 : Float(3, 3) = aten::pow(%a, %2)
#  return (%3)

# 使用torch.jit.script方法进行修饰
# 也可以使用 @torch.jit.script 对函数进行装饰
def func(a):
    if a.norm() > 1.0:
        return a.abs()
    else:
        return a.pow(2)

ret = torch.jit.script(func)
print(ret.graph)
# 打印出的值：
# graph(%a.1 : Tensor):
#   %4 : float = prim::Constant[value=1]()
#   %10 : int = prim::Constant[value=2]()
#   %3 : Tensor = aten::norm(%a.1, %10)
#   %5 : Tensor = aten::gt(%3, %4)
#   %6 : bool = aten::Bool(%5)
#   %18 : Tensor = prim::If(%6)
#     block0():
#       %8 : Tensor = aten::abs(%a.1)
#       -> (%8)
#     block1():
#       %11 : Tensor = aten::pow(%a.1, %10) # -> (%11)
#   return (%18)

class Mod(nn.Module):
    def __init__(self):
        super(Mod, self).__init__()

    # 默认行为: torch.jit.export
    def forward(self, a):
        if a.norm() > 1.0:
            return a.abs()
        else:
            return a.pow(2)

    # 导出该方法
    @torch.jit.export
    def square(self, a):
        return a.pow(2)

    # 不导出该方法
    @torch.jit.ignore
    def abs(self, a):
        return a.abs()

mod = Mod()
ret = torch.jit.script(mod)

