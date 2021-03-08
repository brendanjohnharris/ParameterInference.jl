# ------------------------------------------------------------------------------------------------ #
#                                        Windowing functions                                       #
# ------------------------------------------------------------------------------------------------ #

function slidingWindow(x::Vector, width::Int, overlap::Int, windowFunc::Function=rect)
    # windowFunc must be vectorised
    # Construct sub-timeseries
    X = hcat([x[Array(1:width).+a] for a in 0:(width-overlap):(length(x)-width)]...)
    # Apply the window function to each sub-series
    mapslices(windowFunc, X, dims=1)
end
export slidingWindow




# ------------------------------------------------------------------------------------------------ #
#                                         Window functions                                         #
# ------------------------------------------------------------------------------------------------ #
rect(x::Vector) = x
export rect