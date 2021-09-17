function baselinerotate(dimred)
    function rotate(Fₕ, Fₗ)
        m = project(Array(Fₕ), dimred)
        F -> embed(m, Array(F))
    end
end
export baselinerotate


""""""
function bPCA(Fₕ, Fₗ=zeros(size(Fₕ));
                    transform=identity,
                    prescale=identity,
                    prefilter=identity,
                    rotate=identity,
                    postscale=identity,
                    postfilter=identity)

    transform = transform(Fₕ, Fₗ)
    Fₗ = Fₗ |> transform
    Fₕ = Fₕ |> transform

    prefilter = prefilter(Fₕ, Fₗ)
    Fₗ = Fₗ |> prefilter
    Fₕ = Fₕ |> prefilter

    prescale = prescale(Fₕ, Fₗ)
    Fₗ = Fₗ |> prescale
    Fₕ = Fₕ |> prescale

    rotate = rotate(Fₕ, Fₗ)
    Fₗ = Fₗ |> rotate
    Fₕ = Fₕ |> rotate

    postfilter = postfilter(Fₕ, Fₗ)
    Fₗ = Fₗ |> postfilter
    Fₕ = Fₕ |> postfilter

    postscale = postscale(Fₕ, Fₗ)
    Fₗ = Fₗ |> postscale
    Fₕ = Fₕ |> postscale

    return F -> F |> transform |> prefilter |> prescale |> rotate |> postfilter |> postscale
end
export bPCA

function bPCA(; kwargs...)
    return (x, y) -> bPCA(x, y; kwargs...)
end
