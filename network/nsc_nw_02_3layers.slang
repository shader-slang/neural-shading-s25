// SPDX-License-Identifier: Apache-2.0

import slangpy;

struct NetworkParameters<int Inputs, int Outputs>
{
    Tensor<float, 1> biases;
    Tensor<float, 2> weights;

    // Unlike the mipmapping example, where the parameters for each texel are independent and
    // each texel only affect the material for one sample, the network weights are shared for
    // the entire neural texture. In that case, we need to sum the weight gradients from all
    // the training samples in the batch. We can do this using the atomic tensor
    AtomicTensor<float, 1> biases_grad;
    AtomicTensor<float, 2> weights_grad;

    [Differentiable]
    float get_bias(int neuron)
    {
        return biases.get({neuron});
    }
    [Differentiable]
    float get_weight(int neuron, int input)
    {
        return weights.get({neuron, input});
    }

    [BackwardDerivativeOf(get_bias)]
    void get_bias_bwd(int neuron, float grad)
    {
        // .set on an atomic tensor atomically adds the result
        biases_grad.set({neuron}, grad);
    }

    [BackwardDerivativeOf(get_weight)]
    void get_weight_bwd(int neuron, int input, float grad)
    {
        // .set on an atomic tensor atomically adds the result
        weights_grad.set({neuron, input}, grad);
    }

    // Evaluate all neurons in the layer
    [Differentiable]
    float[Outputs] forward(float[Inputs] x)
    {
        float[Outputs] y;
        [ForceUnroll]
        for (int row = 0; row < Outputs; ++row)
        {
            // Sum up bias plus weighted inputs for 1 neuron
            var sum = get_bias(row);
            [ForceUnroll]
            for (int col = 0; col < Inputs; ++col)
                sum += get_weight(row, col) * x[col];
            y[row] = sum;
        }

        return y;
    }
}

struct Network {
    NetworkParameters<2, 32>  layer0;
    NetworkParameters<32, 32> layer1;
    NetworkParameters<32, 3>  layer2;

    [Differentiable]
    float3 eval(no_diff float2 uv)
    {
        float inputs[2] = {uv.x, uv.y};
        float output0[32] = layer0.forward(inputs);
        [ForceUnroll]
        for (int i = 0; i < 32; ++i)
            output0[i] = activation(output0[i]);
        float output1[32] = layer1.forward(output0);
        [ForceUnroll]
        for (int i = 0; i < 32; ++i)
            output1[i] = activation(output1[i]);
        float output2[3] = layer2.forward(output1);
        [ForceUnroll]
        for (int i = 0; i < 3; ++i)
            output2[i] = activation(output2[i]);
        return float3(output2[0], output2[1], output2[2]);
    }
}

[Differentiable]
float activation(float x)
{
    return max(x, 0.0f);
}

// "Render" the neural texture
[Differentiable]
float3 render(int2 pixel, int2 resolution, Network network)
{
    float2 uv = (float2(pixel) + 0.5f) / float2(resolution);
    return network.eval(uv);
}

[Differentiable]
float3 loss(int2 pixel, int2 resolution, no_diff float3 reference, Network network)
{
    float3 color = render(pixel, resolution, network);
    float3 error = color - reference;
    return error * error; // Squared error
}

struct LCG
{
    uint state;

    __init(uint seed) { state = seed; }

    [mutating]
    uint next_uint()
    {
        const uint A = 1664525u;
        const uint C = 1013904223u;
        state = (A * state + C);
        return state;
    }

    [mutating]
    float next_float()
    {
        // Convert to float in range [0, 1)
        return (next_uint() >> 8) * 0x1p-24;
    }
};

void optimizer_step(inout float primal, inout float grad, inout float mean, inout float variance, float learning_rate, int iteration)
{
    // Default Adam values.
    const float ADAM_BETA_1 = 0.9;
    const float ADAM_BETA_2 = 0.999;
    const float ADAM_EPSILON = 1e-8;

    // Running average of gradient and square gradient (== variance)
    mean = lerp(grad, mean, ADAM_BETA_1);
    variance = lerp(grad * grad, variance, ADAM_BETA_2);

    // The running average is initialized with zeros, and underestimates 
    // the true values at the beginning. Adam divides by a compensation factor
    float mHat = mean / (1.0f - pow(ADAM_BETA_1, iteration));
    float vHat = variance / (1.0f - pow(ADAM_BETA_2, iteration));
    // Compute the adam step by scaling the mean gradient by the learning rate and the variance estimate
    float step = learning_rate * mHat / (sqrt(vHat) + ADAM_EPSILON);

    // Take an adam step
    primal -= step;

    // Because we now accumulate gradients, we need to clear the gradient to zero before the next iteration
    grad = 0.0f;
}

void calculate_grads(uint seed, int2 pixel, int2 resolution, float3 reference, Network network)
{
    bwd_diff(loss)(pixel, resolution, reference, network, 1.0f);
}
