document.querySelector("form").addEventListener("submit", async (event) => {
    event.preventDefault(); // Prevent the form from refreshing the page

    const productId = document.getElementById("product_id").value;

    try {
        const response = await fetch("/recommend", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ product_id: productId }),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        const recommendationsDiv = document.getElementById("recommendations");

        if (data.recommendations && data.recommendations.length > 0) {
            recommendationsDiv.innerHTML = `<h2>Recommendations:</h2><ul>${data.recommendations
                .map((rec) => `<li>${rec}</li>`)
                .join("")}</ul>`;
        } else {
            recommendationsDiv.innerHTML = `<p>No recommendations found for the given Product ID.</p>`;
        }
    } catch (error) {
        console.error("Error fetching recommendations:", error);
        alert("Failed to fetch recommendations. Please try again.");
    }
});
