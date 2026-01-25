using System.Reactive.Disposables;

namespace HOPE.Desktop;

public static class DisposableExtensions
{
    public static T DisposeWith<T>(this T disposable, CompositeDisposable composite) where T : IDisposable
    {
        composite.Add(disposable);
        return disposable;
    }
}
