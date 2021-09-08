function raw(Fₕ, Fₗ=zeros(size(Fₕ)); transform=baselinetransform,
                                             filter=baselinefilter)
    transform = transform(Fₕ, Fₗ)
    Fₗ = Fₗ |> transform
    Fₕ = Fₕ |> transform
    filter = filter(Fₕ, Fₗ)
    Fₗ = Fₗ |> filter
    Fₕ = Fₕ |> filter
    return F -> F |> transform |> filter
end
export raw
