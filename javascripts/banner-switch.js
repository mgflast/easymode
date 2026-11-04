// Swap the header banner: use the Pom banner while browsing the Pom docs section,
// the easymode banner everywhere else. Subscribing to Material's document$ makes this
// run on the initial load and on every instant-navigation page change.
(function () {
  function isHome() {
    var logo = document.querySelector(".md-header__button.md-logo");
    if (!logo) return false;
    var root = new URL(logo.getAttribute("href"), window.location.href).pathname.replace(/index\.html$/, "");
    var here = window.location.pathname.replace(/index\.html$/, "");
    return here === root;
  }
  function update() {
    var isPom = window.location.pathname.indexOf("/pom/") !== -1;
    document.body.classList.toggle("pom-docs", isPom);
    // Tag the landing page so its content can be narrowed and the injected title hidden.
    document.body.classList.toggle("home", isHome());
  }
  if (typeof document$ !== "undefined" && document$.subscribe) {
    document$.subscribe(update);
  } else {
    document.addEventListener("DOMContentLoaded", update);
  }
})();
