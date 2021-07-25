using Plots
using Colors
import PlotThemes._themes
const cornflowerblue = colorant"cornflowerblue"; export cornflowerblue
const crimson = colorant"crimson"; export crimson
const cucumber = colorant"#77ab58"; export cucumber
const california = colorant"#EF9901"; export california
const juliapurple = colorant"#9558b2"; export juliapurple
const keppel = colorant"#46AF98"; export keppel
const darkbg = colorant"#282C34"; export darkbg

function torgba(c::RGB, a::Real=1)
    Colors.RGBA(c.r, c.g, c.b, a)
end

fourseas_palette = torgba.([
    cornflowerblue,
    crimson,
    cucumber,
    california,
    juliapurple
], (0.7,))


PlotThemes._themes[:fourseas] = PlotThemes.PlotTheme(
    foreground_color_text = :black,
    fgguide = :black,
    fglegend = :black,
    legendfontcolor = :black,
    legendtitlefontcolor = :black,
    titlefontcolor = :black,
    palette = fourseas_palette,
    colorgradient = :viridis,
    framestyle = :grid,
    grid = true,
    minorgrid = true,
    minorgridalpha = 1.0,
    foreground_color_minor_grid = :gray91,
    foreground_color_grid = :gray88,
    gridlinewidth = 1.5,
    gridalpha = 1.0,
    titlefontsize = 12,
    tickfontsize = 10,
    legendfontsize = 10,
    legend_titlefontsize = 10,
    fontfamily = "cmu serif",
    minorticks = 2,
); Plots.showtheme(:fourseas)
