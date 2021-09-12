### A Pluto.jl notebook ###
# v0.16.0

using Markdown
using InteractiveUtils

# ╔═╡ 09bee220-1249-11ec-3e2a-4fdde9976a59
begin
    import Pkg
    Pkg.activate(mktempdir())
	Pkg.add(url="https://github.com/brendanjohnharris/Catch22.jl")
	using Catch22
	Pkg.add(url="https://github.com/tk3369/YeoJohnsonTrans.jl")
	Pkg.develop(path="C:\\Users\\Brendan\\OneDrive - The University of Sydney (Students)\\Honours\\Code\\ParameterInference.jl")
	Pkg.add(path="C:\\Users\\Brendan\\OneDrive - The University of Sydney (Students)\\Honours\\Code\\ParameterInference.jl")
	using ParameterInference, Plots
	#plotlyjs();# fourseas!(); # Plots initialisation
end

# ╔═╡ 1cbf17f7-8322-4e3c-a895-163a48332205
md"# Baseline PCA"

# ╔═╡ 2f0cb4e5-4424-42d9-8eb7-e8bf128575ad
md"""
We will construct a dataset that demonstrates the impact of dependencies between features on PCA estimates.
"""

# ╔═╡ 0f10dc15-9153-4d8c-a5e5-c0318c459fa1
md"## Parameters"

# ╔═╡ 230c66e5-7898-48af-9544-439677bee137
md"For a simple example, we will choose to have three 'parameters' that drive our manually constructed 'features': 𝛼, 𝛽 and 𝛾. All will vary as random Gaussians, on the same scale:"

# ╔═╡ fee5a67f-1678-45fc-83b3-3ffc187c19b3
nₛ = 1000

# ╔═╡ 4857cede-b1c9-4a34-99f4-cbb8059111f0
𝛼, 𝛽, 𝛾 = randn(nₛ, 3) |> eachcol

# ╔═╡ 404bb5c9-74ab-43a9-8791-67ac23b0c0f8


# ╔═╡ Cell order:
# ╠═09bee220-1249-11ec-3e2a-4fdde9976a59
# ╠═1cbf17f7-8322-4e3c-a895-163a48332205
# ╠═2f0cb4e5-4424-42d9-8eb7-e8bf128575ad
# ╠═0f10dc15-9153-4d8c-a5e5-c0318c459fa1
# ╠═230c66e5-7898-48af-9544-439677bee137
# ╠═fee5a67f-1678-45fc-83b3-3ffc187c19b3
# ╠═4857cede-b1c9-4a34-99f4-cbb8059111f0
# ╠═404bb5c9-74ab-43a9-8791-67ac23b0c0f8
