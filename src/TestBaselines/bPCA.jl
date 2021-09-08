""""""
function bPCA(Fₕ, Fₗ=zeros(size(Fₕ)); dimred=principalcomponents,
                    transform=baselinetransform,
                    filter=baselinefilter,
                    rotate = (Fₕ, Fₗ) -> F -> embed(project(Fₕ, dimred, pratio=0.99), F), scale=robustintervalscaled)

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
