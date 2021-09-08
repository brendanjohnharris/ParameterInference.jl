function baselinerotate(dimred)
    function rotate(Fₕ, Fₗ)
        m = project(Array(Fₕ), dimred)
        F -> embed(m, Array(F))
    end
end



""""""
function bPCA(Fₕ, Fₗ=zeros(size(Fₕ)); dimred=outlierprincipalcomponents,
                    transform=baselinetransform,
                    filter=baselinefilter,
                    rotate=baselinerotate(dimred),
                    scale=robustintervalscale)

    transform = transform(Fₕ, Fₗ)
    Fₗ = Fₗ |> transform
    Fₕ = Fₕ |> transform

    filter = filter(Fₕ, Fₗ)
    Fₗ = Fₗ |> filter
    Fₕ = Fₕ |> filter

    rotate = rotate(Fₕ, Fₗ)
    Fₗ = Fₗ |> rotate
    Fₕ = Fₕ |> rotate

    scale = scale(Fₕ, Fₗ)

    return F -> F |> transform |> filter |> rotate |> scale
end
export bPCA
