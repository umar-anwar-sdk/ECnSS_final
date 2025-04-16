document.addEventListener("DOMContentLoaded", function () {
  const toast = document.querySelector(".toast"),
        closeIcon = document.querySelector(".close"),
        progress = document.querySelector(".progress");

  let timer1, timer2;

  document.querySelectorAll("button").forEach(button => {
      button.addEventListener("click", () => {
          showToast();
      });
  });

  function showToast() {
      toast.classList.add("active");
      progress.classList.add("active");

      timer1 = setTimeout(() => {
          toast.classList.remove("active");
      }, 5000);

      timer2 = setTimeout(() => {
          progress.classList.remove("active");
      }, 5300);
  }

  if (closeIcon) {
      closeIcon.addEventListener("click", () => {
          toast.classList.remove("active");
          setTimeout(() => progress.classList.remove("active"), 300);
          clearTimeout(timer1);
          clearTimeout(timer2);
      });
  }
});
