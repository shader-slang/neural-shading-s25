# SPDX-License-Identifier: Apache-2.0

from app import App
import slangpy as spy

# Create the app and load the slang module.
app = App(width=1024, height=1024, title="Mipmap Example")
module = spy.Module.load_from_file(app.device, "step_02_mipmap.slang")

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


while app.process_events():

    # Allocate a tensor for output
    output = spy.Tensor.empty_like(albedo_map)

    # Full res rendered output BRDF from full res inputs.
    module.render(
        pixel=spy.call_id(),
        material={"albedo": albedo_map, "normal": normal_map, "roughness": roughness_map},
        light_dir=spy.math.normalize(spy.float3(0.2, 0.2, 1.0)),
        view_dir=spy.float3(0, 0, 1),
        _result=output,
    )

    # Downsample the output tensor.
    output = downsample(output, 2)

    # Blit tensor to screen.
    app.blit(output, size=spy.int2(1024, 1024))

    # Present the window.
    app.present()
