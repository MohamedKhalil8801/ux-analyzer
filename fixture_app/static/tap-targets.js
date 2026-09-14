// Effective tap-target fixture: binds a click handler to the card with
// addEventListener, so the capture's tap instrumentation (which tags
// listener-hosting elements) must measure the card as the tap surface for
// the controls nested inside it.
(function () {
  document.getElementById('card').addEventListener('click', function () {});
})();