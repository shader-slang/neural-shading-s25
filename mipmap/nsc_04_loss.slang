// SPDX-License-Identifier: Apache-2.0

import slangpy;
import brdf;

struct MaterialParameters
{
    Tensor<float3, 2> albedo;
    Tensor<float3, 2> normal;
    Tensor<float, 2> roughness;

    float3 get_albedo(int2 pixel)
    {
        return albedo.getv(pixel);
    }
    float3 get_normal(int2 pixel)
    {
        return normalize(normal.getv(pixel));
    }
    float get_roughness(int2 pixel)
    {
        return roughness.getv(pixel);
    }
};

// Render full res BRDF from given inputs.
float3 render(int2 pixel, MaterialParameters material, float3 light_dir, float3 view_dir)
{
    float light_intensity = 1.5;

    // Sample very shiny BRDF (it rained today!)
    float3 brdf_sample = sample_brdf(
               material.get_albedo(pixel),   // albedo color
               normalize(light_dir),         // light direction
               normalize(view_dir),          // view direction
               material.get_normal(pixel),   // normal map sample
               material.get_roughness(pixel),// roughness
               0.0,                          // metallic (no metal)
               1.0                           // specular
               );

    return brdf_sample * light_intensity;
}

// Render the difference between the rendered pixel a reference pixel.
float3 loss(int2 pixel, float3 reference, MaterialParameters material, float3 light_dir, float3 view_dir)
{
    float3 color = render(pixel, material,
                          light_dir, view_dir);
    float3 error = color - reference;
    return error * error; // Squared error
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
float downsample1(int2 pixel, Tensor<float, 2> source)
{
    float res = 0;
    res += source.getv(pixel * 2 + int2(0, 0));
    res += source.getv(pixel * 2 + int2(1, 0));
    res += source.getv(pixel * 2 + int2(0, 1));
    res += source.getv(pixel * 2 + int2(1, 1));
    return res * 0.25;
}
