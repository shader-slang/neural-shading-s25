// A simple example showing how to use forward and backward mode differentiation
// on a sum of squares function.

// This example is almost the same as square.slang, except that
// the function takes inputs as a custom struct type that is marked
// as differentiable.

struct Point : IDifferentiable
{
    float x;
    float y;
}

[Differentiable]
float square(Point p)
{
    return p.x * p.x + p.y * p.y;
}

void main()
{
    // Forward mode differentiation
    var p = diffPair(Point(3.0, 4.0), Point(1.0, 1.0)); // p = (3, 4), 𝜕p/𝜕𝛉 = (1, 1)
    let result = fwd_diff(square)(p);
    printf("dResult: %f\n", result.d);

    // Backward mode differentiation
    let dLdSquare = 1.0f;
    bwd_diff(square)(p, dLdSquare);
    printf("dL/dx: %f, dL/dy: %f\n", p.d.x, p.d.y);
}
