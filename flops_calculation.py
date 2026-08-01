from thop import profile, clever_format
import models
import torch

def calculate_flops(model_fn=models.hssm, batch_size=1, input_shape=(1, 31, 64, 64)):
    model = model_fn()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    input_tensor = torch.randn(batch_size, *input_shape).to(device)

    macs, params = profile(model, inputs=(input_tensor,), verbose=False)
    macs_str, params_str = clever_format([macs, params], "%.5f")

    # FLOPs = 2 * MACs
    flops_value = float(macs_str.replace('G', '').replace('M', '').replace('K', '')) * 2
    if 'G' in macs_str:
        print(f"Total FLOPs: {flops_value:.2f}G")
    elif 'M' in macs_str:
        print(f"Total FLOPs: {flops_value:.2f}M")
    else:
        print(f"Total FLOPs: {flops_value:.2f}K")
    print(f"Total MACs: {macs_str}")
    print(f"Total Parameters: {params_str}")
    print("--------------------------------")


calculate_flops(models.hssm, batch_size=1, input_shape=(1, 31, 64, 64))
calculate_flops(models.hcanet, batch_size=1, input_shape=(1, 31, 64, 64))
# calculate_flops(models.munet, batch_size=1, input_shape=(1, 31, 64, 64))
calculate_flops(models.trq3d, batch_size=1, input_shape=(1, 31, 64, 64))
# calculate_flops(models.qrnn3d, batch_size=1, input_shape=(1, 31, 64, 64))
# calculate_flops(models.sert_base, batch_size=1, input_shape=(1, 31, 64, 64))
# calculate_flops(models.sst, batch_size=1, input_shape=(1, 31, 64, 64))
# calculate_flops(models.trq3d, batch_size=1, input_shape=(1, 31, 64, 64))
# calculate_flops(models.hsd, batch_size=1, input_shape=(1, 31, 64, 64))

# Example Usage:
# calculate_flops(models.hssm, batch_size=1, input_shape=(1, 31, 64, 64))


# model = models.memnet()
# batch_size = 1
# input_shape = (batch_size, 31, 64, 64)

# flops, macs, params = calculate_flops(model=model,
#                                       input_shape=input_shape,
#                                       output_as_string=False,
#                                       output_precision=4,
#                                       print_results=True,
#                                       print_detailed=False)
# print("MemNet FLOPs:%4.2fG   MACs:%4.2fM   Params:%3.2fM \n" %(flops/(10**9), macs/(10**9), params/(10**6)))

# model = models.hsidcnn()
# from flopth import flopth
# flops, params = flopth(model, in_size=((1, 64, 64), (1, 31, 64, 64)))
# print("HSIDCNN FLOPs:%s  Params:%s \n" %(flops, params))

# model = models.smcnn()
# from flopth import flopth
# flops, params = flopth(model, in_size=((1, 64, 64), (1, 31, 64, 64)))
# print("SMCNN FLOPs:%s  Params:%s \n" %(flops, params))


# model = models.grunet()
# batch_size = 1
# input_shape = (batch_size, 1, 31, 64, 64)

# flops, macs, params = calculate_flops(model=model,
#                                       input_shape=input_shape,
#                                       output_as_string=False,
#                                       output_precision=4,
#                                       print_results=True,
#                                       print_detailed=False)
# print("GRUNET FLOPs:%4.2fG   MACs:%4.2fM   Params:%3.2fM \n" %(flops/(10**9), macs/(10**9), params/(10**6)))

# model = models.qrnn3d()
# batch_size = 1
# input_shape = (batch_size, 1, 31, 64, 64)

# flops, macs, params = calculate_flops(model=model,
#                                       input_shape=input_shape,
#                                       output_as_string=False,
#                                       output_precision=4,
#                                       print_results=True,
#                                       print_detailed=False)
# print("QRNN3D FLOPs:%4.2fG   MACs:%4.2fM   Params:%3.2fM \n" %(flops/(10**9), macs/(10**9), params/(10**6)))

# model = models.sert_base()
# batch_size = 1
# input_shape = (batch_size, 31, 64, 64)

# flops, macs, params = calculate_flops(model=model,
#                                       input_shape=input_shape,
#                                       output_as_string=False,
#                                       output_precision=4,
#                                       print_results=True,
#                                       print_detailed=False)
# print("SERT FLOPs:%4.2fG   MACs:%4.2fM   Params:%3.2fM \n" %(flops/(10**9), macs/(10**9), params/(10**6)))
# #Alexnet FLOPs:4.2892 GFLOPS   MACs:2.1426 GMACs   Params:61.1008 M

# model = models.sst()
# batch_size = 1
# input_shape = (batch_size, 31, 64, 64)

# flops, macs, params = calculate_flops(model=model,
#                                       input_shape=input_shape,
#                                       output_as_string=False,
#                                       output_precision=4,
#                                       print_results=True,
#                                       print_detailed=False)
# print("SST FLOPs:%4.2fG   MACs:%4.2fM   Params:%3.2fM \n" %(flops/(10**9), macs/(10**9), params/(10**6)))

# model = models.trq3d()
# batch_size = 1
# input_shape = (batch_size, 1, 31, 64, 64)

# flops, macs, params = calculate_flops(model=model,
#                                       input_shape=input_shape,
#                                       output_as_string=False,
#                                       output_precision=4,
#                                       print_results=True,
#                                       print_detailed=False)
# print("TRQ3D FLOPs:%4.2fG   MACs:%4.2fM   Params:%3.2fM \n" %(flops/(10**9), macs/(10**9), params/(10**6)))


# model = models.hsdt()
# batch_size = 1
# input_shape = (batch_size, 1, 31, 64, 64)

# flops, macs, params = calculate_flops(model=model,
#                                       input_shape=input_shape,
#                                       output_as_string=False,
#                                       output_precision=4,
#                                       print_results=True,
#                                       print_detailed=False)
# print("HSDT FLOPs:%4.2fG   MACs:%4.2fM   Params:%3.2fM \n" %(flops/(10**9), macs/(10**9), params/(10**6)))


# model = models.hsdt_new24csp()
# batch_size = 1
# input_shape = (batch_size, 1, 31, 64, 64)

# flops, macs, params = calculate_flops(model=model,
#                                       input_shape=input_shape,
#                                       output_as_string=False,
#                                       output_precision=4,
#                                       print_results=True,
#                                       print_detailed=False)
# print("CST3D_24 FLOPs:%4.2fG   MACs:%4.2fM   Params:%3.2fM \n" %(flops/(10**9), macs/(10**9), params/(10**6)))


# model = models.hsdt_new16csp()
# batch_size = 1
# input_shape = (batch_size, 1, 31, 64, 64)

# flops, macs, params = calculate_flops(model=model,
#                                       input_shape=input_shape,
#                                       output_as_string=False,
#                                       output_precision=4,
#                                       print_results=True,
#                                       print_detailed=False)
# print("CST3D_16 FLOPs:%4.2fG   MACs:%4.2fM   Params:%3.2fM \n" %(flops/(10**9), macs/(10**9), params/(10**6)))


# model = models.hsdt_new8csp()
# batch_size = 1
# input_shape = (batch_size, 1, 31, 64, 64)

# flops, macs, params = calculate_flops(model=model,
#                                       input_shape=input_shape,
#                                       output_as_string=False,
#                                       output_precision=4,
#                                       print_results=True,
#                                       print_detailed=False)
# print("CST3D_8 FLOPs:%4.2fG   MACs:%4.2fM   Params:%3.2fM \n" %(flops/(10**9), macs/(10**9), params/(10**6)))

# model = models.hsdt_new4csp()
# batch_size = 1
# input_shape = (batch_size, 1, 31, 64, 64)

# flops, macs, params = calculate_flops(model=model,
#                                       input_shape=input_shape,
#                                       output_as_string=False,
#                                       output_precision=4,
#                                       print_results=True,
#                                       print_detailed=False)
# print("CST3D_4 FLOPs:%4.2fG   MACs:%4.2fM   Params:%3.2fM \n" %(flops/(10**9), macs/(10**9), params/(10**6)))

# model = models.man()
# batch_size = 1
# input_shape = (batch_size, 1, 31, 64, 64)

# flops, macs, params = calculate_flops(model=model,
#                                       input_shape=input_shape,
#                                       output_as_string=False,
#                                       output_precision=4,
#                                       print_results=True,
#                                       print_detailed=False)
# print("MAN FLOPs:%4.2fG   MACs:%4.2fM   Params:%3.2fM \n" %(flops/(10**9), macs/(10**9), params/(10**6)))

# model = models.hdnet()
# batch_size = 1
# input_shape = (batch_size, 31, 64, 64)

# flops, macs, params = calculate_flops(model=model,
#                                       input_shape=input_shape,
#                                       output_as_string=False,
#                                       output_precision=4,
#                                       print_results=True,
#                                       print_detailed=False)
# print("HDNET FLOPs:%4.2fG   MACs:%4.2fM   Params:%3.2fM \n" %(flops/(10**9), macs/(10**9), params/(10**6)))

