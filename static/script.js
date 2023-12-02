$( document ).ready(() => {
    $("#Header").append(`
        <nav class="navbar navbar-expand-lg navbar-light bg-light">
            <a class="navbar-brand" href="/">Neural Network Designer</a>
            <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav">
                    <li class="nav-item active">
                        <a class="nav-link" href="/">Home <span class="sr-only">(current)</span></a>
                    </li>
                    <!-- <li class="nav-item">
                        <a class="nav-link" href="/network">Networks</a>
                    </li> -->
                </ul>
            </div>
        </nav>`
    );

    $("#Footer").append(`
        <footer style="">
            <h2>Cheeky ;)</h2>
        </footer>`
    );


    $("#Footer").css("width", "100%");
    $("#Footer").css("text-align", "center");
    $("#Footer").css("justify-content", "center");
    $("#Footer").css("display", "flex");
    $("#Footer").css("padding", "1rem");
    $("#Footer").css("bottom", "0");
});

$("body").resizer(() => {
    if ($("body").height() < $(window).height()){
        $("#Footer").css("position", "absolute");
    }
    else {
        $("#Footer").css("position", "relative");
    }
})
$(window).on('resize', () => {
    if ($("body").height() < $(window).height()){
        $("#Footer").css("position", "absolute");
    }
    else {
        $("#Footer").css("position", "relative");
    }
})