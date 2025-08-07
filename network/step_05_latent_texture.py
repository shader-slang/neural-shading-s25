# SPDX-License-Identifier: Apache-2.0

from app import App
import slangpy as spy
import numpy as np

# Create the app and load the slang module.
app = App(width=512 * 3 + 10 * 2, height=512, title="Network Example")
module = spy.Module.load_from_file(app.device, "step_05_latent_texture.slang")

# Load some materials.
image = spy.Tensor.load_from_image(app.device, "slangstars.png", linearize=True)


class NetworkParameters(spy.InstanceList):
    def __init__(self, inputs: int, outputs: int):
        super().__init__(module[f"NetworkParameters<{inputs},{outputs}>"])
        self.inputs = inputs
        self.outputs = outputs

        # Biases and weights for the layer.
        self.biases = spy.Tensor.from_numpy(app.device, np.zeros(outputs).astype("float32"))
        self.weights = spy.Tensor.from_numpy(
            app.device, np.random.uniform(-0.5, 0.5, (outputs, inputs)).astype("float32")
        )

        # Gradients for the biases and weights.
        self.biases_grad = spy.Tensor.zeros_like(self.biases)
        self.weights_grad = spy.Tensor.zeros_like(self.weights)

        # Temp data for Adam optimizer.
        self.m_biases = spy.Tensor.zeros_like(self.biases)
        self.m_weights = spy.Tensor.zeros_like(self.weights)
        self.v_biases = spy.Tensor.zeros_like(self.biases)
        self.v_weights = spy.Tensor.zeros_like(self.weights)

    # Calls the Slang 'optimize' function for biases and weights
    def optimize(self, learning_rate: float, optimize_counter: int):
        module.optimizer_step(
            self.biases,
            self.biases_grad,
            self.m_biases,
            self.v_biases,
            learning_rate,
            optimize_counter,
        )
        module.optimizer_step(
            self.weights,
            self.weights_grad,
            self.m_weights,
            self.v_weights,
            learning_rate,
            optimize_counter,
        )


class LatentTexture(spy.InstanceList):
    def __init__(self, width: int, height: int, num_latents: int):
        super().__init__(module[f"LatentTexture<{num_latents}>"])
        self.width = width
        self.height = height
        self.num_latents = num_latents
        
        # Initialize to random latent texture
        initial_latents = np.random.uniform(0.0, 1.0, (height, width, num_latents)).astype("float32")
        self.texture = spy.Tensor.from_numpy(app.device, initial_latents)

        # Gradients for the latent texture
        self.texture_grads = spy.Tensor.zeros_like(self.texture)

        # Temp data for Adam optimizer.
        self.m_texture = spy.Tensor.zeros_like(self.texture)
        self.v_texture = spy.Tensor.zeros_like(self.texture)

    # Calls the Slang 'optimize' function for biases and weights
    def optimize(self, learning_rate: float, optimize_counter: int):
        module.optimizer_step(
            self.texture,
            self.texture_grads,
            self.m_texture,
            self.v_texture,
            learning_rate,
            optimize_counter,
        )


class Network(spy.InstanceList):
    def __init__(self):
        super().__init__(module["Network"])
        self.latent_texture = LatentTexture(32, 32, 4)
        self.layer0 = NetworkParameters(4, 32)
        self.layer1 = NetworkParameters(32, 32)
        self.layer2 = NetworkParameters(32, 3)

    # Calls the Slang 'optimize' function for the layer.
    def optimize(self, learning_rate: float, optimize_counter: int):
        self.latent_texture.optimize(learning_rate, optimize_counter)
        self.layer0.optimize(learning_rate, optimize_counter)
        self.layer1.optimize(learning_rate, optimize_counter)
        self.layer2.optimize(learning_rate, optimize_counter)


network = Network()

optimize_counter = 0

# Slang will compile the shaders the first time we call into them (i.e. in the first iteration)
print("Compiling shaders... this can take a while")

while app.process_events():

    offset = 0
    # Blit reference texture
    app.blit(image, size=spy.int2(512), offset=spy.int2(offset, 0), tonemap=False, bilinear=True)
    offset += 512 + 10
    res = spy.int2(256, 256)

    # Render current neural texture
    lr_output = spy.Tensor.empty_like(image)
    module.render(pixel=spy.call_id(), resolution=res, network=network, _result=lr_output)
    app.blit(lr_output, size=spy.int2(512, 512), offset=spy.int2(offset, 0), tonemap=False, bilinear=True)
    offset += 512 + 10

    # Show loss between neural texture and reference texture.
    loss_output = spy.Tensor.empty_like(image)
    module.loss(
        pixel=spy.call_id(), resolution=res, network=network, reference=image, _result=loss_output
    )
    app.blit(loss_output, size=spy.int2(512, 512), offset=spy.int2(offset, 0), tonemap=False)

    learning_rate = 0.001

    for i in range(50):
        module.calculate_grads(
            seed=spy.wang_hash(seed=optimize_counter, warmup=2),
            pixel=spy.call_id(),
            resolution=res,
            reference=image,
            network=network,
        )
        optimize_counter += 1

        network.optimize(learning_rate, optimize_counter)

    print(f"Loss: {np.mean(loss_output.to_numpy()):.5f}")

    # Present the window.
    app.present()
