""""""
function intervalscaled(Fₕ, Fₗ=zeros(size(Fₕ)); transform=baselinetransform(Fₕ, Fₗ),
    filter=baselinefilter(Fₕ, Fₗ))
    Fₗ = Fₗ |> transform |> filter
    Fₕ = Fₕ |> transform |> filter
    scale = intervalscale(Fₗ, Fₕ, rampOn) # So, extrapolate if variance is greater than high dim.
    return F -> F |> transform |> filter |> scale
end
export intervalscaled
