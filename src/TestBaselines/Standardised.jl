""""""
function standardised(unused...; transform=baselinetransform,
    filter=baselinefilter)
    function f(F)
        Fₗ = zeros(size(F))
        Fₕ = F # No baseline, so the best we can do is treat the test as its own baseline
        intervalscaled(Fₕ, Fₗ)(F)
    end
    return f
end
export standardised
