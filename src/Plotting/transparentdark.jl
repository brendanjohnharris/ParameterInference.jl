using PlotThemes
using Colors
import PlotThemes._themes
# See:
#    https://medialab.github.io/iwanthue/
# and:
#    https://chir.ag/projects/name-that-color/
const cornflowerblue = colorant"cornflowerblue"; export cornflowerblue
const crimson = colorant"crimson"; export crimson
const cucumber = colorant"#77ab58"; export cucumber
const copper = colorant"#c37940"; export copper
const fuchsia = colorant"#be5ea7"; export fuschia
const dark = colorant"#282C34"; export dark

dark_palette = [
    cornflowerblue,
    crimson,
    cucumber,
    copper
]

# Not actually transparent, for some reason that default doesn't propagate
_themes[:transparentdark] = PlotThemes.PlotTheme(
    bg = colorant"#282C34",
    bginside = colorant"#282C34",
    fg = colorant"#FFFFFF",
    fg_axis = colorant"#FFFFFF",
    foreground_color_border = colorant"#FFFFFF",
    foreground_color_text = colorant"#FFFFFF",
    fgtext = colorant"#FFFFFF",
    fgguide = colorant"#FFFFFF",
    fglegend = colorant"#FFFFFF",
    legendfontcolor = colorant"#FFFFFF",
    legendtitlefontcolor = colorant"#FFFFFF",
    titlefontcolor = colorant"#FFFFFF",
    framestyle = :box,
    palette = PlotThemes.expand_palette(RGB(0, 0, 0), dark_palette; lchoices = [57], cchoices = [100]),
    colorgradient = :viridis
)
