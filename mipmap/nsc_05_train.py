# SPDX-License-Identifier: Apache-2.0

import math

import numpy as np
from app import App
import slangpy as spy

# Create the app and load the slang module.
app = App(width=3092, height=1024, title="Mipmap Example")
module = spy.Module.load_from_file(app.device, "nsc_05_train.slang")

# Load some materials.
albedo_map = spy.Tensor.load_from_image(
    app.device, "PavingStones070_2K.diffuse.jpg", linearize=True
)
normal_map = spy.Tensor.load_from_image(
    app.device, "PavingStones070_2K.normal.jpg", scale=2, offset=-1
)
roughness_map = spy.Tensor.load_from_image(
    app.device, "PavingStones070_2K.roughness.jpg", grayscale=True
)


def downsample(source: spy.Tensor, steps: int) -> spy.Tensor:
    for i in range(steps):
        dest = spy.Tensor.empty(
            device=app.device,
            shape=(source.shape[0] // 2, source.shape[1] // 2),
            dtype=source.dtype,
        )
        if dest.dtype.name == "vector":
            module.downsample3(spy.call_id(), source, _result=dest)
        else:
            module.downsample1(spy.call_id(), source, _result=dest)
        source = dest
    return source


# Downsampled maps
lr_albedo_map = downsample(albedo_map, 2)
lr_normal_map = downsample(normal_map, 2)
lr_roughness_map = downsample(roughness_map, 2)

lr_trained_albedo_map = spy.Tensor.zeros_like(lr_albedo_map)
lr_trained_normal_map = spy.Tensor.zeros_like(lr_normal_map)
lr_trained_roughness_map = spy.Tensor.zeros_like(lr_roughness_map)

module.init3(lr_trained_albedo_map, spy.float3(0.5, 0.5, 0.5))
module.init_normal(lr_trained_normal_map)
module.init1(lr_trained_roughness_map, 0.5)

# Corresponding gradients
lr_albedo_grad = spy.Tensor.zeros_like(lr_albedo_map)
lr_normal_grad = spy.Tensor.zeros_like(lr_normal_map)
lr_roughness_grad = spy.Tensor.zeros_like(lr_roughness_map)

m_albedo = spy.Tensor.zeros_like(lr_albedo_grad)
v_albedo = spy.Tensor.zeros_like(lr_albedo_grad)
m_normal = spy.Tensor.zeros_like(lr_normal_grad)
v_normal = spy.Tensor.zeros_like(lr_normal_grad)
m_roughness = spy.Tensor.zeros_like(lr_roughness_grad)
v_roughness = spy.Tensor.zeros_like(lr_roughness_grad)


def getRandomDir():
    r = math.sqrt(np.random.rand())
    phi = np.random.rand() * math.pi * 2
    Lx = r * math.sin(phi)
    Ly = r * math.cos(phi)
    Lz = math.sqrt(max(1 - r**2, 0))
    return spy.float3(Lx, Ly, Lz)


optimize_counter = 0

while app.process_events():

    light_dir = spy.math.normalize(spy.float3(0.2, 0.2, 1.0))
    xpos = 0
    bilinear_output = True

    # Full res rendered output BRDF from full res inputs.
    output = spy.Tensor.empty_like(albedo_map)
    module.render(
        pixel=spy.call_id(),
        material={
            "albedo": albedo_map,
            "normal": normal_map,
            "roughness": roughness_map,
        },
        light_dir=light_dir,
        view_dir=spy.float3(0, 0, 1),
        _result=output,
    )

    # Downsample the output tensor.
    output = downsample(output, 2)

    # Blit tensor to screen.
    app.blit(output, size=spy.int2(1024, 1024), offset=spy.int2(xpos, 0), bilinear=bilinear_output)
    xpos += 1024 + 10

    # Same but using trained normal map res rendered output BRDF from quarter res inputs.
    lr_output = spy.Tensor.empty_like(output)
    module.render(
        pixel=spy.call_id(),
        material={
            "albedo": lr_trained_albedo_map,
            "normal": lr_trained_normal_map,
            "roughness": lr_trained_roughness_map,
        },
        light_dir=light_dir,
        view_dir=spy.float3(0, 0, 1),
        _result=lr_output,
    )

    # Blit tensor to screen.
    app.blit(
        lr_output, size=spy.int2(1024, 1024), offset=spy.int2(xpos, 0), bilinear=bilinear_output
    )
    xpos += 1024 + 10

    # Loss between downsampled output and quarter res rendered output.
    orig_loss_output = spy.Tensor.empty_like(output)
    module.loss(
        pixel=spy.call_id(),
        material={
            "albedo": lr_albedo_map,
            "normal": lr_normal_map,
            "roughness": lr_roughness_map,
        },
        reference=output,
        light_dir=light_dir,
        view_dir=spy.float3(0, 0, 1),
        _result=orig_loss_output,
    )

    # Loss between downsampled output and quarter res rendered output.
    loss_output = spy.Tensor.empty_like(output)
    module.loss(
        pixel=spy.call_id(),
        material={
            "albedo": lr_trained_albedo_map,
            "normal": lr_trained_normal_map,
            "roughness": lr_trained_roughness_map,
        },
        reference=output,
        light_dir=light_dir,
        view_dir=spy.float3(0, 0, 1),
        _result=loss_output,
    )

    # Blit tensor to screen.
    app.blit(
        loss_output, size=spy.int2(1024, 1024), offset=spy.int2(xpos, 0), tonemap=bilinear_output
    )
    xpos += 1024 + 10

    # Extra credit: Start with a fast learning rate and slowly ramp down
    training_progress_percentage = min(optimize_counter / 3000, 1.0)
    learning_rate = (
        0.002 * (1.0 - training_progress_percentage) + 0.0002 * training_progress_percentage
    )

    # Loss between downsampled output and quarter res rendered output.
    # NOTE: iterations can be turned up to 1000 if you want to see results quicker!
    for i in range(50):
        module.calculate_grads(
            seed=spy.wang_hash(seed=optimize_counter, warmup=2),
            pixel=spy.grid(shape=lr_albedo_map.shape),
            material={
                "albedo": lr_trained_albedo_map,
                "normal": lr_trained_normal_map,
                "roughness": lr_trained_roughness_map,
                "albedo_grad": lr_albedo_grad,
                "normal_grad": lr_normal_grad,
                "roughness_grad": lr_roughness_grad,
            },
            ref_material={
                "albedo": albedo_map,
                "normal": normal_map,
                "roughness": roughness_map,
            },
        )
        optimize_counter += 1

        # Optimize the trained maps using the gradients.
        module.optimizer_step3(
            lr_trained_albedo_map,
            lr_albedo_grad,
            m_albedo,
            v_albedo,
            learning_rate,
            optimize_counter,
            False,
        )
        module.optimizer_step3(
            lr_trained_normal_map,
            lr_normal_grad,
            m_normal,
            v_normal,
            learning_rate,
            optimize_counter,
            True,
        )
        module.optimizer_step1(
            lr_trained_roughness_map,
            lr_roughness_grad,
            m_roughness,
            v_roughness,
            learning_rate,
            optimize_counter,
        )

    # read loss output to numpy tensor and sum abs values
    orig_loss_np = orig_loss_output.to_numpy()
    orig_loss_value = np.sum(np.abs(orig_loss_np))
    loss_np = loss_output.to_numpy()
    loss_value = np.sum(np.abs(loss_np))
    print(f"Loss: {loss_value:.6f}, Original Loss: {orig_loss_value:.6f}")

    # Present the window.
    app.present()
