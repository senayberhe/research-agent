export function ComingSoon({ title }: { title: string }) {
  return (
    <>
      <div className="page-header">
        <h1>{title}</h1>
      </div>
      <p className="coming-soon">This page is next on the list.</p>
    </>
  );
}
