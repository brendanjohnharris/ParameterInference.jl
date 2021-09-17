# ------------------------------------------------------------------------------------------------ #
#                                        Windowing functions                                       #
# ------------------------------------------------------------------------------------------------ #

function slidingWindow(x::AbstractVector, width::Int=length(x)÷20, overlap::Int=0, windowFunc::Function=rect)
    # windowFunc must be vectorised
    # Construct sub-timeseries
    X = hcat([x[Array(1:width).+a] for a in 0:(width-overlap):(length(x)-width)]...)
    idxs = 1:size(X, 1):(size(X, 2)*size(X, 1)+1)
    # Apply the window function to each sub-series
    (mapslices(windowFunc, X, dims=1), idxs)
end
function slidingWindow(width::Int, args...)
    f(x) = slidingWindow(x, width, args...)
end
export slidingWindow

function slidingwindow(x::AbstractVector, width::Int=length(x)÷20, overlap::Int=0, windowFunc::Function=rect)
    # windowFunc must be vectorised
    # Construct sub-timeseries
    X = hcat([x[Array(1:width).+a] for a in 0:(width-overlap):(length(x)-width)]...)
    idxs = 1:size(X, 1):(size(X, 2)*size(X, 1)+1)
    # Apply the window function to each sub-series
    mapslices(windowFunc, X, dims=1)
end
function slidingwindow(width::Int, args...)
    f(x) = slidingwindow(x, width, args...)
end
export slidingwindow




# ------------------------------------------------------------------------------------------------ #
#                                         Window functions                                         #
# ------------------------------------------------------------------------------------------------ #
rect(x::AbstractVector) = x
export rect
