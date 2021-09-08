""""""
function bPCA(Fₕ, Fₗ=zeros(size(Fₕ)); dimred=principalcomponents, transform=baselinetransform, filter=baselinefilter)

    transform = transform(Fₕ, Fₗ)
    Fₗ = Fₗ |> transform
    Fₕ = Fₕ |> transform
    filter = filter(Fₕ, Fₗ)
    Fₗ = Fₗ |> filter
    Fₕ = Fₕ |> filter

    scale1 = intervalscaled(Fₕ, Fₗ; transform=identity, filter=identity)
    Fₗ = Fₗ |> scale1
    Fₕ = Fₕ |> scale1

    m = project(Fₕ, dimred, pratio=0.99)
    function rotation(F)
        embed(m, F)
    end
    Fₗ = Fₗ |> rotation
    Fₕ = Fₕ |> rotation
    scale = intervalscaled(Fₕ, Fₗ; transform=identity, filter=identity)
    return F -> F |> transform |> filter |> scale1 |> rotation |> scale
end
export bPCA
