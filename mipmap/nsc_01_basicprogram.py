# SPDX-License-Identifier: Apache-2.0

from app import App
import slangpy as spy

# Create the app and load the slang module.
app = App(width=2048, height=2048, title="Mipmap Example")
module = spy.Module.load_from_file(app.device, "nsc_01_basicprogram.slang")

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

    # Blit tensor to screen.
    app.blit(output)

    # Present the window.
    app.present()
