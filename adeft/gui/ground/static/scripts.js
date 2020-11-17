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
	    grounding = event.target.getAttribute('data-grounding');
	    split_grounding = grounding.split(':')
	    if (split_grounding.length == 3) {
		namespace = split_grounding[0];
		identifier = split_grounding[1] + ':' + split_grounding[2];
	    }
	    else if (split_grounding.length == 2) {
		namespace = split_grounding[0];
		identifier = split_grounding[1];
	    }
	    else if (split_grounding.length == 1) {
		namespace = '';
		identifier = split_grounding[0];
	    }
	    else {
		namespace = '';
		identifier = '';
	    }
	    document.getElementById('namespace-box').value = namespace;
	    document.getElementById('identifier-box').value = identifier;
	});
    }
}
