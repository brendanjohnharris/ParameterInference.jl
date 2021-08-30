"""Standardising is just assuming the zero-dim variance is 0"""
function standardised(Fₕ, unused...; transform=baselinetransform(Fₕ, zeros(size(Fₕ))),
    filter=baselinefilter(Fₕ, zeros(size(Fₕ))))
    Fₗ=zeros(size(Fₕ))
    Fₗ = Fₗ |> transform |> filter
    Fₕ = Fₕ |> transform |> filter
    scale = intervalscale(Fₗ, Fₕ, rampOn) # So, extrapolate if variance is greater than high dim.
    return F -> F |> transform |> filter |> scale
end
