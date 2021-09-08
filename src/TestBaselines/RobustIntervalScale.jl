""""""
function robustintervalscaled(Fₕ, Fₗ=zeros(size(Fₕ)); transform=baselinetransform,
    filter=baselinefilter)
    transform = transform(Fₕ, Fₗ)
    Fₗ = Fₗ |> transform
    Fₕ = Fₕ |> transform
    filter = filter(Fₕ, Fₗ)
    Fₗ = Fₗ |> filter
    Fₕ = Fₕ |> filter
    scale = intervalscale(Fₗ, Fₕ, rampOn) # So, extrapolate if variance is greater than high dim.
    return F -> F |> transform |> filter |> scale
end
export robustintervalscaled
