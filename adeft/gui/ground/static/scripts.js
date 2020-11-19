document.addEventListener("DOMContentLoaded", function (event) {
    var scrollpos = sessionStorage.getItem('scrollpos');
    if (scrollpos) {
        window.scrollTo(0, scrollpos);
        sessionStorage.removeItem('scrollpos');
    }
    var namespaceBoxContent = sessionStorage.getItem('namespace-box-content');
    var nameBoxContent = sessionStorage.getItem('name-box-content');
    var identifierBoxContent = sessionStorage.getItem('identifier-box-content');
    if (namespaceBoxContent) {
	document.getElementById('namespace-box').value = namespaceBoxContent;
	sessionStorage.removeItem('namespace-box-content');
    }
    if (nameBoxContent) {
	document.getElementById('name-box').value = nameBoxContent;
	sessionStorage.removeItem('name-box-content');
    }
    if (identifierBoxContent) {
	document.getElementById('identifier-box').value = identifierBoxContent;
	sessionStorage.removeItem('identifier-box-content');
    }
});

function setScrollPosition() {
    sessionStorage.setItem('scrollpos', window.scrollY);
}

function setScrollPositionAndInputBoxContent() {
    sessionStorage.setItem('scrollpos', window.scrollY);
    sessionStorage.setItem('namespace-box-content',
			   document.getElementById('namespace-box').value);
    sessionStorage.setItem('name-box-content',
			   document.getElementById('name-box').value);
    sessionStorage.setItem('identifier-box-content',
			   document.getElementById('identifier-box').value);
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
		var namespace = split_grounding[0];
		var identifier = split_grounding[1] + ':' + split_grounding[2];
	    }
	    else if (split_grounding.length == 2) {
		var namespace = split_grounding[0];
		var identifier = split_grounding[1];
	    }
	    else if (split_grounding.length == 1) {
		var namespace = '';
		var identifier = split_grounding[0];
	    }
	    else {
		var namespace = '';
		var identifier = '';
	    }
	    document.getElementById('namespace-box').value = namespace;
	    document.getElementById('identifier-box').value = identifier;
	});
    }
}
