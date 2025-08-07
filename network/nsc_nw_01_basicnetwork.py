# SPDX-License-Identifier: Apache-2.0

from app import App
import slangpy as spy
import numpy as np

# Create the app and load the slang module.
app = App(width=512 * 3 + 10 * 2, height=512, title="Network Example")
module = spy.Module.load_from_file(app.device, "nsc_nw_01_basicnetwork.slang")

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


class Network(spy.InstanceList):
    def __init__(self):
        super().__init__(module["Network"])
        self.layer = NetworkParameters(2, 3)

    # Calls the Slang 'optimize' function for the layer.
    def optimize(self, learning_rate: float, optimize_counter: int):
        self.layer.optimize(learning_rate, optimize_counter)


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
