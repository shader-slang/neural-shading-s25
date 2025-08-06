// SPDX-License-Identifier: Apache-2.0

import slangpy;
import brdf;

struct MaterialParameters
{
    RWTensor<float3, 2> albedo;
    RWTensor<float3, 2> normal;
    RWTensor<float, 2> roughness;

    RWTensor<float3, 2> albedo_grad;
    RWTensor<float3, 2> normal_grad;
    RWTensor<float, 2> roughness_grad;

    [Differentiable]
    float3 get_albedo(int2 pixel)
    {
        return albedo.getv(pixel);
    }

    [BackwardDerivativeOf(get_albedo)]
    void get_albedo_bwd(int2 pixel, float3 grad)
    {
        albedo_grad.setv(pixel, grad);
    }

    [Differentiable]
    float3 get_normal(int2 pixel)
    {
        return normal.getv(pixel);
    }

    [BackwardDerivativeOf(get_normal)]
    void get_normal_bwd(int2 pixel, float3 grad)
    {
        normal_grad.setv(pixel, grad);
    }

    [Differentiable]
    float get_roughness(int2 pixel)
    {
        return roughness.getv(pixel);
    }
    [BackwardDerivativeOf(get_roughness)]
    void get_roughness_bwd(int2 pixel, float grad)
    {
        roughness_grad.setv(pixel, grad);
    }
};

// Render full res BRDF from given inputs.
[Differentiable]
float3 render(int2 pixel, no_diff MaterialParameters material, no_diff float3 light_dir, no_diff float3 view_dir)
{
    float light_intensity = 1.5;

    // Sample very shiny BRDF (it rained today!)
    float3 brdf_sample = sample_brdf(
               material.get_albedo(pixel),  // albedo color
               normalize(light_dir),        // light direction
               normalize(view_dir),         // view direction
               normalize(material.get_normal(pixel)),  // normal map sample
               material.get_roughness(pixel), // roughness
               0.0,                         // metallic (no metal)
               1.0                          // specular
               );

    return brdf_sample * light_intensity;
}

// Render the difference between the rendered pixel a reference pixel.
[Differentiable]
float3 loss(int2 pixel, no_diff float3 reference, no_diff MaterialParameters material, no_diff float3 light_dir, no_diff float3 view_dir)
{
    float3 color = render(pixel, material,
                          light_dir, view_dir);
    float3 error = color - reference;
    return error * error; // Squared error
}

// LCG based random number generator that can give us
// uniformly distributed random numbers on a hemisphere.
#define PI 3.14159265358979323846f
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

    [mutating] float3 next_dir_on_hemisphere()
    {
        // Random azimuthal angle [0, 2π)
        float phi = 2.0f * PI * next_float();

        // Random polar angle [0, π/2) for hemisphere
        float cos_theta = next_float();
        float sin_theta = sqrt(1.0f - cos_theta * cos_theta);

        // Convert to Cartesian coordinates
        return float3(sin_theta * cos(phi),sin_theta * sin(phi),cos_theta);
    }
};

void calculate_grads(uint seed, int2 pixel, MaterialParameters material, MaterialParameters ref_material)
{
    LCG lcg = LCG(seed);

    // Sample light and view direction from a hemisphere.
    float3 light_dir = lcg.next_dir_on_hemisphere();
    float3 view_dir = lcg.next_dir_on_hemisphere();

    // Index of high res pixel we'll sample.
    int2 hi_res_pixel = pixel * 4;

    // Render and average 16 samples from the high resolution material.
    float3 sum = 0;
    for(int x0 = 0; x0 < 4; x0++)
    {
        for(int y0 = 0; y0 < 4; y0++)
        {
            int2 hi_res_pixel_offset = hi_res_pixel + int2(x0, y0);
            sum += render(hi_res_pixel_offset, ref_material, light_dir, view_dir);
        }
    }
    sum /= 16.0f;

    // Back-propagate through the loss function.
    // results in get_albedo_bwd, get_normal_bwd, get_roughness_bwd being
    // called with the gradients, which are then stored in buffers.
    bwd_diff(loss)(pixel, sum, material, light_dir, view_dir, 1);
}

void init_normal(inout float3 val)
{
    val = float3(0,0,1);
}
void init3(inout float3 val, float3 v)
{
    val = v;
}
void init1(inout float val, float v)
{
    val = v;
}

void adam_step(inout float primal, float grad, inout float mean, inout float variance, float learning_rate, int iteration)
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
}

void optimizer_step1(inout float primal, inout float grad, inout float mean, inout float variance, float learning_rate, int iteration)
{
    adam_step(primal, grad, mean, variance, learning_rate, iteration);

    // Make sure we stay in [0, 1]
    primal = saturate(primal);
}

void optimizer_step3(inout float3 primal, float3 grad, inout float3 mean, inout float3 variance, float learning_rate, int iteration, bool renormalize)
{
    for (int i = 0; i < 3; ++i)
        adam_step(primal[i], grad[i], mean[i], variance[i], learning_rate, iteration);

    if (renormalize)
        primal = normalize(primal); // If it's a normal, make sure it's still normalized.
    else
        primal = saturate(primal); // ...otherwise, make sure we stay in [0, 1]
}

float downsample1(int2 pixel, Tensor<float, 2> source)
{
    float res = 0;
    res += source.getv(pixel * 2 + int2(0, 0));
    res += source.getv(pixel * 2 + int2(1, 0));
    res += source.getv(pixel * 2 + int2(0, 1));
    res += source.getv(pixel * 2 + int2(1, 1));
    return res * 0.25;
}

float3 downsample3(int2 pixel, Tensor<float3, 2> source)
{
    float3 res = 0;
    res += source.getv(pixel * 2 + int2(0, 0));
    res += source.getv(pixel * 2 + int2(1, 0));
    res += source.getv(pixel * 2 + int2(0, 1));
    res += source.getv(pixel * 2 + int2(1, 1));
    return res * 0.25;
}
