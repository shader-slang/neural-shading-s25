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
