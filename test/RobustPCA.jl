using ParameterInference
using LowRankModels

function rprincipalcomponents(A::AbstractArray, k::Int, scale::Float64=1.0; abs_tol=1e-4^3, rel_tol=1e-4^3, max_iter=10000, kwargs...)
	loss = L1Loss()
	r = ZeroReg()
    # # Start with PCA
    m = GLRM(A, QuadLoss(), ZeroReg(), ZeroReg(), k; kwargs...)
    fit!(m)
    # Then do the other thing
    m = GLRM(A, loss, r, r, k; kwargs...)
    fit!(m, ProxGradParams(;abs_tol, rel_tol, max_iter))
	return m
end



function nanprincipalcomponents(A::AbstractArray, k::Int, scale::Float64=1.0; abs_tol=1e-4^3, rel_tol=1e-4^3, max_iter=10000, kwargs...)
    A = SparseArrays.fkeep!(A, (i,j,x) -> abs(x) < 5) # ! better
	loss = QuadLoss()
	r = ZeroReg()
    m = GLRM(A, loss, r, r, k; kwargs...)
    fit!(m, ProxGradParams(;abs_tol, rel_tol, max_iter))
	return m
end
