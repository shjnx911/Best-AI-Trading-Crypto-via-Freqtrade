const { createChart } = require('lightweight-charts');
const methods = Object.getOwnPropertyNames(createChart.prototype);
console.log('Available methods:', methods);
