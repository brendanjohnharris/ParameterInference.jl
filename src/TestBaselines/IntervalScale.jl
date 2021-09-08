""""""
function intervalscaled(Fₕ, Fₗ=zeros(size(Fₕ)))
    scale = intervalscale(Fₗ, Fₕ, rampOn) # So, extrapolate if variance is greater than high dim.
    return F -> F |> scale
end
export intervalscaled
