// A simple example showing how to use forward and backward mode differentiation
// on a sum of squares function.

[Differentiable]
float square(float x, float y)
{
    return x * x + y * y;
}

void main()
{
    // Forward mode differentiation
    var x = diffPair(3.0, 1.0); // x = 3, 𝜕x/𝜕𝛉 = 1
    var y = diffPair(4.0, 1.0); // y = 4, 𝜕y/𝜕𝛄 = 1
    let result = fwd_diff(square)(x, y);
    printf("dResult: %f\n", result.d);

    // Backward mode differentiation
    let dLdSquare = 1.0f;
    bwd_diff(square)(x, y, dLdSquare);
    printf("dL/dx: %f, dL/dy: %f\n", x.d, y.d);
}
