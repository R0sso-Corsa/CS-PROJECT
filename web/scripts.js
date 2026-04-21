document.addEventListener("DOMContentLoaded", () => {
    const searchInput = document.querySelector('input[name="ticker"]');
    const fillButtons = document.querySelectorAll("[data-fill-ticker]");

    fillButtons.forEach((button) => {
        button.addEventListener("click", () => {
            if (searchInput) {
                searchInput.value = button.getAttribute("data-fill-ticker") || "";
                searchInput.focus();
            }
        });
    });
});
