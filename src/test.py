





def third_order_delta(gradients, h=1e-5):
    # 计算 f(x+2h)
    f_x_plus_2h = gradients + 2 * h
    # 计算 f(x+h)
    f_x_plus_h = gradients + h
    # 计算 f(x-h)
    f_x_minus_h = gradients - h
    # 计算 f(x-2h)
    f_x_minus_2h = gradients - 2 * h
    
    # 计算三阶导数近似
    third_order = (f_x_plus_2h - 2 * f_x_plus_h + 2 * f_x_minus_h - f_x_minus_2h) / (2 * h**3)
    
    return third_order
