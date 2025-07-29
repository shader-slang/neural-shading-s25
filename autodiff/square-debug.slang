// A simple example derived from square.slang, with the addition to use
// custom derivatives to print gradients of intermediate terms for debugging.

[Differentiable]
float square(float x, float y)
{
    let x2 = x * x;
    let y2 = debugGrad("y2", y * y);
    return x2 + y2;
}

float debugGrad(NativeString s, float x) { return x; }
[BackwardDerivativeOf(debugGrad)]
void debugGradBwd(NativeString s, inout DifferentialPair<float> x, float dOut)
{
    printf("Gradient of %s is %f\n", s, dOut);
    x = diffPair(x.p, dOut);
}

void main()
{
    // Backward mode differentiation
    var x = diffPair(3.0, 0.0);
    var y = diffPair(4.0, 0.0);
    let dLdSquare = 1.0f;
    bwd_diff(square)(x, y, dLdSquare);
    printf("dL/dx: %f, dL/dy: %f\n", x.d, y.d);
}
