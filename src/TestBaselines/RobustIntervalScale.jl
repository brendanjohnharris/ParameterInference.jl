""""""
function robustintervalscale(Fₕ, Fₗ=zeros(size(Fₕ)))
    σₗ = mapslices(iqr, Fₗ, dims=2)
    σₕ = mapslices(iqr, Fₕ, dims=2)
    function scale(F)
        σ = mapslices(iqr, F, dims=2)
        σᵢ = rampOn(0, 1, σₗ, σₕ).(σ) # No need to worry about means?
        return F.*σᵢ./σ
    end
    return F -> F |> scale
end
export robustintervalscaled
