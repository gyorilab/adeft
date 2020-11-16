document.addEventListener("DOMContentLoaded", function (event) {
    var scrollpos = sessionStorage.getItem('scrollpos');
    if (scrollpos) {
        window.scrollTo(0, scrollpos);
        sessionStorage.removeItem('scrollpos');
    }
});

function setScrollPosition() {
  sessionStorage.setItem('scrollpos', window.scrollY);
}


window.onload = function() {
    groundings = document.getElementsByClassName('click-grounded');
    for (var i = 0; i < groundings.length; i++) {
	groundings[i].addEventListener('click', function (event) {
	    document.getElementById('name-box').value =
		event.target.getAttribute("data-name");
	    document.getElementById('grounding-box').value =
		event.target.getAttribute('data-grounding');
	});
    }
}
