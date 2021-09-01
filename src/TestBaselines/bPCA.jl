""""""
function bPCA(Fₕ, Fₗ=zeros(size(Fₕ)); dimred=principalcomponents, transform=baselinetransform(Fₕ, Fₗ), filter=baselinefilter(Fₕ, Fₗ))
    filter=baselinefilter(Fₕ, Fₗ)
    Fₗ = Fₗ |> transform |> filter
    Fₕ = Fₕ |> transform |> filter
    m = project(Fₕ, dimred, pratio=0.95)
    rotation(F) = embed(m, F)
    Fₗ = Fₗ |> rotation
    Fₕ = Fₕ |> rotation
    scale = intervalscaled(Fₕ, Fₗ; transform=identity, filter=identity)
    return F -> F |> filter |> rotation |> scale
end
export bPCA
