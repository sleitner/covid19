/**
 * Detects if WebGL is enabled.
 * Inspired from http://www.browserleaks.com/webgl#howto-detect-webgl
 *
 * @return { number } -1 for not Supported,
 *                    0 for disabled
 *                    1 for enabled
 */
function detectWebGL()
{
    // Check for the WebGL rendering context
    if ( !! window.WebGLRenderingContext) {
        var canvas = document.createElement("canvas"),
            names = ["webgl", "experimental-webgl", "moz-webgl", "webkit-3d"],
            context = false;

        for (var i in names) {
            try {
                context = canvas.getContext(names[i]);
                if (context && typeof context.getParameter === "function") {
                    // WebGL is enabled.
                    return 'enabled';
                }
            } catch (e) {}
        }

        // WebGL is supported, but disabled.
        return 'disabled';
    }

    // WebGL not supported.
    return 'no support';
};
