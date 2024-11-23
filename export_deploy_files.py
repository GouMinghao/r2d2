from r2d2.tools.load_net import load_network
from r2d2.tools.load_image import load_image

import torch
import os
import matplotlib.pyplot as plt

if __name__ == "__main__":
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    net = load_network("models/faster2d2_WASF_N8_big.pt")
    img_tensor = load_image(
        "/home/gmh/git/nn_deploy/third_party/r2d2/imgs/brooklyn.png"
    )
    res = net(img_tensor)

    desc = res[0].detach().numpy()
    rep = res[1].detach().numpy()
    rel = res[2].detach().numpy()

    plt.figure()
    plt.jet()
    plt.subplot(1, 2, 1)
    plt.title("rel")
    plt.imshow(rel[0, 0, :, :])
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.title("rep")
    plt.imshow(rep[0, 0, :, :])
    plt.savefig(os.path.join(output_dir, "rel_rep_vis.png"))

    desc.tofile(os.path.join(output_dir, "desc.bin"))
    rep.tofile(os.path.join(output_dir, "rep.bin"))
    rel.tofile(os.path.join(output_dir, "rel.bin"))

    traced_module = torch.jit.trace(net, img_tensor)
    traced_module.save(os.path.join(output_dir, "r2d2_traced_torch_script.pt"))
    onnx_model = torch.onnx.export(
        net, (img_tensor,), f=os.path.join(output_dir, "r2d2.onnx")
    )
