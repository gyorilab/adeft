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
