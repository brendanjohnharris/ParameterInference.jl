function raw(Fₕ, Fₗ=zeros(size(Fₕ)); transform=baselinetransform(Fₕ, Fₗ),
                                             filter=baselinefilter(Fₕ, Fₗ))
    return F -> F |> transform |> filter
end
export raw
