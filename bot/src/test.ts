const hash =
	'BC5HgSRgKfvHc1HSinaaqUxNE2faQXtvcGnp29eyYSBqxmjxULdteCCVgBgT3SLZXijX9e8eyuEBC6XCV1gHhuq';
const confirmResponse = await fetch(
	'https://swift.drift.trade/confirmation/hash-status?hash=' +
		encodeURIComponent(hash)
);

console.log(confirmResponse);
